# We need to set this variable to shut-up TensorFlow's C++ logging messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import importlib
import tensorflow_datasets.public_api as tfds

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
flags.DEFINE_boolean('do_test', False, 'Evaluate the performance of a pretrained model')
flags.DEFINE_integer('max_seq_len', 128, 'Maximum sequence length')
flags.DEFINE_string('model_name', 'bert-base-cased', 'Name of pretrained transformer model to load')
flags.DEFINE_string('checkpoint_file', None, 'Path to save checkpoints')
flags.DEFINE_string('data_dir', None, 'Path to TensorFlow DataSet home (e.g., ~/tensorflow_datasets)')
flags.DEFINE_string('cache_dir', None, 'Path to save TensorFlow DataSet cache files (e.g., /tmp)')
flags.DEFINE_string('checksum_dir', '/data/LHC_kitchensink/tensorflow_datasets/url_checksums',
                    help='Path to checksum directory')
flags.DEFINE_integer('steps_per_epoch', 100, 'Number of steps considered as an epoch')
flags.DEFINE_enum('implementation', default='tensorflow', enum_values=['tensorflow', 'pytorch'],
                  help='implementation to use for huggingface models')
flags.DEFINE_enum('evaluation', default='basic', enum_values=['basic', 'nlg'],
                  help='method to use for evaluating model performance')


# noinspection PyUnusedLocal
def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    experiment: experiments.Experiment
    if FLAGS.implementation == 'tensorflow':
        # configure_tf(FLAGS.use_xla, FLAGS.use_amp)
        experiment = experiments.TFExperiment(tokenizer_name=FLAGS.model_name,
                                              data_dir=FLAGS.data_dir,
                                              max_seq_len=FLAGS.max_seq_len,
                                              use_xla=FLAGS.use_xla,
                                              use_amp=FLAGS.use_amp)
    elif FLAGS.implementation == 'pytorch':
        experiment = experiments.PTExperiment(tokenizer_name=FLAGS.model_name,
                                              data_dir=FLAGS.data_dir,
                                              max_seq_len=FLAGS.max_seq_len,
                                              use_amp=FLAGS.use_amp,
                                              warmup_epochs=FLAGS.warmup_epochs)
    else:
        raise NotImplementedError('Unsupported implementation \"%s\"' % FLAGS.implementation)

    # Load model
    model = experiment.load_model(model_name=FLAGS.model_name)
   
    if FLAGS.do_train:
        # Parse dataset and split
        training_tasks = [experiments.Task.parse(task) for task in FLAGS.training_tasks]
        validation_tasks = [experiments.Task.parse(task) for task in FLAGS.validation_tasks]

        # Train model
        logging.info('Training %s with %s...', FLAGS.model_name, ' '.join(FLAGS.training_tasks))
        experiment.train(model,
                         training_tasks=training_tasks,
                         validation_tasks=training_tasks,
                         num_epochs=FLAGS.num_epochs,
                         steps_per_epoch=FLAGS.steps_per_epoch,
                         prefetch_size=FLAGS.prefetch_size,
                         batch_size=FLAGS.batch_size,
                         eval_batch_size=FLAGS.eval_batch_size,
                         eval_batches=FLAGS.eval_batches,
                         checkpoint_file=FLAGS.checkpoint_file)

    if FLAGS.do_test:
        # Evaluate the model
        testing_tasks = [experiments.Task.parse(task) for task in FLAGS.testing_tasks]
        logging.info('Evaluating %s with %s...', FLAGS.model_name, ' '.join(FLAGS.testing_tasks))
        predictions = experiment.predict(model,
                                         tasks=testing_tasks,
                                         eval_batch_size=FLAGS.eval_batch_size,
                                         eval_batches=FLAGS.eval_batches,
                                         splits=[tfds.Split.VALIDATION, tfds.Split.TEST])

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

    if not FLAGS.do_test and not FLAGS.do_train:
       raise ValueError('Please specify training and/or testing mode.')


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
