# We need to set this variable to shut-up TensorFlow's C++ logging messages
import csv
import os
import faulthandler
import signal
import sys
from typing import Dict

faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

import numpy as np

import gorilla

from fslks.experiments import Predictions, Task

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow_datasets.core.utils import gcs_utils

from absl import flags
from absl import app
from absl import logging

from fslks import tasks
from fslks import experiments
from fslks import evaluation

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("training_tasks", [], "One or more tasks to be used for pretraining")
flags.DEFINE_spaceseplist("validation_tasks", [], "One or more tasks to be used for validation during pretraining")
flags.DEFINE_spaceseplist("testing_tasks", [], "One or more tasks to be used for evaluating pretrained models")

flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train')
flags.DEFINE_integer('warmup_epochs', 3, 'Number of warmup epochs before normal training')
flags.DEFINE_integer('batch_size', 8, 'Batch size to use for training')
flags.DEFINE_integer('prefetch_size', -1, 'Number of batches to prefetch (default: AUTOTUNE)')
flags.DEFINE_integer('eval_batch_size', 8, 'Batch size to use when evaluating validation/test sets')
flags.DEFINE_integer('eval_batches', 10, 'Number of batches to evaluate when testing')
flags.DEFINE_boolean('use_xla', False, 'Enable XLA optimization')
flags.DEFINE_boolean('use_amp', False, 'Enable AMP optimization')
flags.DEFINE_boolean('do_train', False, 'Train and validate the specified model')
flags.DEFINE_boolean('do_predict', False, 'Save (trained) model predictions model')
flags.DEFINE_boolean('do_test', False, 'Evaluate the performance of a (trained) model')
flags.DEFINE_integer('max_seq_len', 512, 'Maximum sequence length')
flags.DEFINE_string('init_checkpoint', 't5-base', 'Name of pretrained transformer model to load')
flags.DEFINE_string('checkpoint_dir', None, 'Path to save checkpoints')
flags.DEFINE_string('prediction_dir', None, 'Path to save/load predictions')
flags.DEFINE_string('data_dir', None, 'Path to TensorFlow DataSets home (e.g., ~/tensorflow_datasets)')
flags.DEFINE_string('cache_dir', None, 'Path to save TensorFlow DataSet cache files (e.g., /tmp)')
flags.DEFINE_string('checksum_dir', '/data/LHC_kitchensink/tensorflow_datasets/url_checksums',
                    help='Path to checksum directory')
flags.DEFINE_integer('steps_per_epoch', 1000, 'Number of steps considered as an epoch')
flags.DEFINE_enum('implementation', default='pytorch', enum_values=['tensorflow', 'pytorch'],
                  help='implementation to use for huggingface models')
flags.DEFINE_enum('evaluation', default='basic', enum_values=['basic', 'nlg'],
                  help='method to use for evaluating model performance')
flags.DEFINE_integer('seed', default=1337, help='Random seed used for experiments')
flags.DEFINE_float('temperature', default=2., help='Temperature used for task mixing')
flags.DEFINE_boolean('dynamic_mixing', default=False,
                     help='Whether to turn on dynamic task mixing based on validation losses')
flags.DEFINE_boolean('mix_from_validation', default=True,
                     help='If True, dynamic mixing will use validation losses; otherwise, training losses will be used.')
flags.DEFINE_float('clip_mixing_size', default=2e14, help='Maximum size to clip datasets for proprtional mixing')
flags.DEFINE_integer('test_limit', default=sys.maxsize, help='Maximum number of predictions to evaluate per task')


def save_predictions(predictions: Predictions, output_dir: str):
    logging.info('Saving predictions for tasks %s', set(predictions.keys()))
    for task, task_predictions in predictions.items():
        logging.info('Saving predictions for %s splits %s', task, set(task_predictions.keys()))
        for split, split_predictions_fn in task_predictions.items():
            split_predictions = split_predictions_fn()
            if split_predictions is not None:
                split_output_dir = os.path.join(output_dir, task, str(split))
                os.makedirs(split_output_dir, exist_ok=True)
                output_file = os.path.join(split_output_dir, 'predictions.csv')
                logging.info('Saving %d predictions for %s[%s] at %s...', len(split_predictions['prompt']), task, split,
                             output_file)
                with open(output_file, 'w') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['prompt', 'predictions', 'targets'])
                    for prompt_, predictions_, targets_ in zip(split_predictions['prompt'],
                                                               split_predictions['predictions'],
                                                               split_predictions['targets']):
                        writer.writerow([prompt_, predictions_, targets_])


def load_predictions(output_dir: str, testing_tasks) -> Predictions:
    predictions: Predictions = {}
    for task in testing_tasks:
        predictions_file = os.path.join(output_dir, task.dataset, str(task.split), 'predictions.csv')
        if not os.path.exists(predictions_file):
            logging.warning('Unable to load predictions for %s: %s not found', task, predictions_file)
            continue

        split_predictions = []
        targets = []
        prompts = []
        with open(predictions_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                split_predictions.append(row['predictions'])
                targets.append(row['targets'])
                prompts.append(row['prompt'])

        if task not in predictions:
            predictions[task.dataset] = {}

        # Python does lazy binding so we need to store the results in an immutable variable, and then
        # use the variable as the default argument to the lambda since the default argument is actually
        # eagerly bound when the lambda is declared. Yes, this is awful.
        results = {
            'prompts': np.asarray(prompts),
            'predictions': np.asarray(split_predictions),
            'targets': np.asarray(targets)
        }
        # noinspection PyDefaultArgument
        predictions[task.dataset][task.split] = lambda t=results: t

        logging.info('Loaded %d predictions for %s', len(prompts), task)
    return predictions


# noinspection PyUnusedLocal
def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    if FLAGS.do_train or FLAGS.do_predict or (FLAGS.do_test and not FLAGS.prediction_dir):
        experiment: experiments.Experiment
        if FLAGS.implementation == 'tensorflow':
            # configure_tf(FLAGS.use_xla, FLAGS.use_amp)
            experiment = experiments.TFExperiment(cache_dir=FLAGS.cache_dir,
                                                  configuration_name=FLAGS.init_checkpoint,
                                                  max_seq_len=FLAGS.max_seq_len,
                                                  use_xla=FLAGS.use_xla,
                                                  use_amp=FLAGS.use_amp,
                                                  seed=FLAGS.seed)
        elif FLAGS.implementation == 'pytorch':
            experiment = experiments.PTExperiment(cache_dir=FLAGS.cache_dir,
                                                  configuration_name=FLAGS.init_checkpoint,
                                                  max_seq_len=FLAGS.max_seq_len,
                                                  use_amp=FLAGS.use_amp,
                                                  warmup_epochs=FLAGS.warmup_epochs,
                                                  seed=FLAGS.seed,
                                                  temperature=FLAGS.temperature,
                                                  dynamic_mixing=FLAGS.dynamic_mixing,
                                                  mix_from_validation=FLAGS.mix_from_validation,
                                                  clip_mixing_size=FLAGS.clip_mixing_size)
        else:
            raise NotImplementedError('Unsupported implementation \"%s\"' % FLAGS.implementation)

        # Load model
        model = experiment.load_model(model_name=FLAGS.init_checkpoint)

    patch_settings = gorilla.Settings(allow_hit=True)

    def _patched_gcs_dataset_info_files(dataset_dir):
        try:
            original = gorilla.get_original_attribute(gcs_utils, 'gcs_dataset_info_files')
            return original(dataset_dir)
        except IOError as ioe:
            logging.error('Failed to connect to GCS', exc_info=ioe)
            return None

    patch = gorilla.Patch(gcs_utils, 'gcs_dataset_info_files', _patched_gcs_dataset_info_files, settings=patch_settings)
    gorilla.apply(patch)

    # Setup tfds parameters
    Task.data_dir = FLAGS.data_dir
    Task.add_checksum_dir(FLAGS.checksum_dir)

    # Register all our defined task mappings
    tasks.register_task_mappings()

    if FLAGS.do_train:
        # Parse dataset and split
        training_tasks = Task.parse_train_tasks(FLAGS.training_tasks)
        validation_tasks = Task.parse_validation_tasks(FLAGS.validation_tasks)

        if FLAGS.dynamic_mixing and FLAGS.mix_from_validation:
            train_sets: Dict[str, Task] = {t.dataset: t for t in training_tasks}
            valid_sets: Dict[str, Task] = {t.dataset: t for t in validation_tasks}
            if train_sets.keys() != valid_sets.keys():
                logging.error('Dynamic mixing from validation requites validation data for each training task!')
            for dataset in train_sets.keys() - valid_sets.keys():
                if Task.split_in_dataset("validation", dataset):
                    valid_sets[dataset] = Task(dataset, 'validation')
                    logging.warning('Adding %s to validation tasks', dataset)
                else:
                    train_sets[dataset] = Task(dataset, 'train[:70%]')
                    valid_sets[dataset] = Task(dataset, 'train[-30%:]')
                    logging.warning('Adjusting %s to use 80%% for training and 20%% for validation', dataset)
            training_tasks = []
            validation_tasks = []
            for dataset in train_sets:
                training_tasks.append(train_sets[dataset])
                validation_tasks.append(valid_sets[dataset])
            for dataset in valid_sets.keys() - train_sets.keys():
                validation_tasks.append(valid_sets[dataset])

        if FLAGS.checkpoint_dir:
            # Make directories to save best checkpoint and final checkpoint
            os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
            FLAGS.append_flags_into_file(os.path.join(FLAGS.checkpoint_dir, 'flags.cfg'))
            best_dir = "{0}_best".format(FLAGS.checkpoint_dir)
            os.makedirs(best_dir, exist_ok=True)
            FLAGS.append_flags_into_file(os.path.join(best_dir, 'flags.cfg'))

        # Train model
        logging.info('Training %s with %s...', FLAGS.init_checkpoint, ' '.join(FLAGS.training_tasks))
        experiment.train(model,
                         training_tasks=training_tasks,
                         validation_tasks=validation_tasks,
                         num_epochs=FLAGS.num_epochs,
                         steps_per_epoch=FLAGS.steps_per_epoch,
                         prefetch_size=FLAGS.prefetch_size,
                         batch_size=FLAGS.batch_size,
                         eval_batch_size=FLAGS.eval_batch_size,
                         eval_batches=FLAGS.eval_batches,
                         checkpoint_file=FLAGS.checkpoint_dir)

        if FLAGS.checkpoint_dir:
            # Save final checkpoint
            experiment.save_model(model, FLAGS.checkpoint_dir)

    if FLAGS.do_predict:
        # Evaluate the model
        testing_tasks = Task.parse_test_tasks(FLAGS.testing_tasks)
        # Reload model, using best checkpoint if available.
        # Otherwise use the existing model.
        model_dir = "{0}_best".format(FLAGS.checkpoint_dir)
        if os.path.isdir(model_dir):
            logging.info("Loading best performing checkpoint: %s" % (model_dir))
            experiment.load_model(model_name=model_dir)

        logging.info('Predicting %s with %s...', ' '.join(FLAGS.testing_tasks), FLAGS.init_checkpoint)

        predictions = experiment.predict(model,
                                         tasks=testing_tasks,
                                         eval_batch_size=FLAGS.eval_batch_size)
        save_predictions(predictions, FLAGS.prediction_dir)

    if FLAGS.do_test:
        testing_tasks = Task.parse_test_tasks(FLAGS.testing_tasks)
        if FLAGS.prediction_dir:
            predictions = load_predictions(FLAGS.prediction_dir, testing_tasks)
        else:
            logging.warning('--prediction_dir was not specified, generating predictions from scratch')
            predictions = experiment.predict(model,
                                             tasks=testing_tasks,
                                             eval_batch_size=FLAGS.eval_batch_size)

        evaluator = evaluation.get_evaluator(FLAGS.evaluation)
        results = evaluator.evaluate(predictions, FLAGS.test_limit)
        print('Results:')
        print(results)

    if not any([FLAGS.do_train, FLAGS.do_predict, FLAGS.do_test]):
        logging.error('Please specify at least one of --do_train, --do_predict, or --do_test')


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
