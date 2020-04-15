# We need to set this variable to shut-up TensorFlow's C++ logging messages
import csv
import os

import numpy as np

from fslks.experiments import Predictions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib

from absl import flags
from absl import app
from absl import logging

# We need to import our custom TensorFlow DataSet Builders
# noinspection PyUnresolvedReferences
from fslks import tasks
from fslks import experiments
from fslks import evaluation

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("training_tasks", None, "One or more tasks to be used for pretraining")
flags.DEFINE_spaceseplist("validation_tasks", None, "One or more tasks to be used for validation during pretraining")
flags.DEFINE_spaceseplist("testing_tasks", None, "One or more tasks to be used for evaluating pretrained models")

flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train')
flags.DEFINE_integer('warmup_epochs', 3, 'Number of warmup epochs before normal training')
flags.DEFINE_integer('batch_size', 128, 'Batch size to use for training')
flags.DEFINE_integer('prefetch_size', 10, 'Number of batches to prefetch')
flags.DEFINE_integer('eval_batch_size', 128, 'Batch size to use when evaluating validation/test sets')
flags.DEFINE_integer('eval_batches', 100, 'Number of batches to evaluate when testing')
flags.DEFINE_boolean('use_xla', False, 'Enable XLA optimization')
flags.DEFINE_boolean('use_amp', False, 'Enable AMP optimization')
flags.DEFINE_boolean('do_train', False, 'Train and validate the specified model')
flags.DEFINE_boolean('do_predict', False, 'Save (trained) model predictions model')
flags.DEFINE_boolean('do_test', False, 'Evaluate the performance of a (trained) model')
flags.DEFINE_integer('max_seq_len', 128, 'Maximum sequence length')
flags.DEFINE_string('init_checkpoint', 'bert-base-cased', 'Name of pretrained transformer model to load')
flags.DEFINE_string('checkpoint_dir', None, 'Path to save checkpoints')
flags.DEFINE_string('prediction_dir', None, 'Path to save/load predictions')
flags.DEFINE_string('data_dir', None, 'Path to TensorFlow DataSet home (e.g., ~/tensorflow_datasets)')
flags.DEFINE_string('cache_dir', None, 'Path to save TensorFlow DataSet cache files (e.g., /tmp)')
flags.DEFINE_string('checksum_dir', '/data/LHC_kitchensink/tensorflow_datasets/url_checksums',
                    help='Path to checksum directory')
flags.DEFINE_integer('steps_per_epoch', 100, 'Number of steps considered as an epoch')
flags.DEFINE_enum('implementation', default='tensorflow', enum_values=['tensorflow', 'pytorch'],
                  help='implementation to use for huggingface models')
flags.DEFINE_enum('evaluation', default='basic', enum_values=['basic', 'nlg'],
                  help='method to use for evaluating model performance')


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
    for task, split in testing_tasks:
        if split is None:
            split = "test"
        predictions_file = os.path.join(output_dir, task, split, 'predictions.csv')
        if not os.path.exists(predictions_file):
            logging.warning('Unable to load predictions for %s[%s]: %s not found', task, split, predictions_file)
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
            predictions[task] = {}

        # Python does lazy binding so we need to store the results in an immutable variable, and then
        # use the variable as the default argument to the lambda since the default argument is actually
        # eagerly bound when the lambda is declared. Yes, this is awful.
        results = {
            'prompts': np.asarray(prompts),
            'predictions': np.asarray(split_predictions),
            'targets': np.asarray(targets)
        }
        predictions[task][split] = lambda t=results: t

        logging.info('Loaded %d predictions for %s[%s]', len(prompts), task, split)
    return predictions


# noinspection PyUnusedLocal
def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    experiment: experiments.Experiment
    if FLAGS.implementation == 'tensorflow':
        # configure_tf(FLAGS.use_xla, FLAGS.use_amp)
        experiment = experiments.TFExperiment(tokenizer_name=FLAGS.init_checkpoint,
                                              data_dir=FLAGS.data_dir,
                                              max_seq_len=FLAGS.max_seq_len,
                                              use_xla=FLAGS.use_xla,
                                              use_amp=FLAGS.use_amp)
    elif FLAGS.implementation == 'pytorch':
        experiment = experiments.PTExperiment(tokenizer_name=FLAGS.init_checkpoint,
                                              data_dir=FLAGS.data_dir,
                                              max_seq_len=FLAGS.max_seq_len,
                                              use_amp=FLAGS.use_amp,
                                              warmup_epochs=FLAGS.warmup_epochs)
    else:
        raise NotImplementedError('Unsupported implementation \"%s\"' % FLAGS.implementation)

    # Load model
    model = experiment.load_model(model_name=FLAGS.init_checkpoint)

    if FLAGS.do_train:
        # Parse dataset and split
        training_tasks = [experiments.Task.parse(task) for task in FLAGS.training_tasks]
        validation_tasks = [experiments.Task.parse(task) for task in (FLAGS.validation_tasks or FLAGS.training_tasks)]

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
            try:
                os.mkdir(FLAGS.checkpoint_dir)
            except FileExistsError:
                pass
            experiment.save_model(model, FLAGS.checkpoint_dir)

    if FLAGS.do_predict:
        # Evaluate the model
        testing_tasks = [experiments.Task.parse(task) for task in FLAGS.testing_tasks]
        logging.info('Evaluating %s with %s...', FLAGS.init_checkpoint, ' '.join(FLAGS.testing_tasks))
        predictions = experiment.predict(model,
                                         tasks=testing_tasks,
                                         eval_batch_size=FLAGS.eval_batch_size,
                                         eval_batches=FLAGS.eval_batches)
        save_predictions(predictions, FLAGS.prediction_dir)

    if FLAGS.do_test:
        testing_tasks = [experiments.Task.parse(task) for task in FLAGS.testing_tasks]
        predictions = load_predictions(FLAGS.prediction_dir, testing_tasks)
        logging.info('Results:')
        evaluator: evaluation.Evaluator
        if FLAGS.evaluation == 'basic':
            evaluator = evaluation.BasicEvaluator()
        elif FLAGS.evaluation == 'nlg':
            nlg_eval = importlib.import_module('nlgeval')
            evaluator = evaluation.NlgEvaluator(nlg=nlg_eval.NLGEval())
        else:
            raise NotImplementedError('Unsupported evaluator \"' + FLAGS.evaluation + "\"")

        results = evaluator.evaluate(predictions)
        print(results)

    if not any([FLAGS.do_train, FLAGS.do_predict, FLAGS.do_test]):
        logging.error('Please specify at least one of --do_train, --do_predict, or --do_test')


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
