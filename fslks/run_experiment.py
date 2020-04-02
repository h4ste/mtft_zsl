import os

# Make TensorFlow print less obnoxious C logging messages
# (must be set before tensorflow is imported!)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from absl import flags
from absl import app
from absl import logging

# We need to import our custom TensorFlow DataSet Builders
# noinspection PyUnresolvedReferences
from fslks import tasks
from fslks import experiments

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("tasks", None, "One or more tasks to be used for pretraining")

flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 128, 'Batch size to use for training')
flags.DEFINE_integer('prefetch_size', 10, 'Number of batches to prefetch')
flags.DEFINE_integer('eval_batch_size', 128, 'Batch size to use when evaluating validation/test sets')
flags.DEFINE_integer('eval_batches', 100, 'Number of batches to evaluate when testing')
flags.DEFINE_boolean('use_xla', False, 'Enable XLA optimization')
flags.DEFINE_boolean('use_amp', False, 'Enable AMP optimization')
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


def configure_tf(use_xla: bool = False,
                 use_amp: bool = False) -> None:
    logging.info(('Enabling' if use_xla else 'Disabling') + ' XLA optimization')
    tf.config.optimizer.set_jit(use_xla)
    logging.info(('Enabling' if use_amp else 'Disabling') + ' auto mixed precision (AMP)')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_amp})


def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    experiment: experiments.Experiment
    if FLAGS.implementation == 'tensorflow':
        configure_tf(FLAGS.use_xla, FLAGS.use_amp)
        experiment = experiments.TFExperiment(tokenizer_name=FLAGS.model_name,
                                              data_dir=FLAGS.data_dir,
                                              max_seq_len=FLAGS.max_seq_len)
    elif FLAGS.implementation == 'pytorch':
        # When you're ready, uncomment these lines (assuming your pytorch experiment class is named PTExperiment)
        # experiment = experiments.PTExperiment(tokenizer_name=FLAGS.model_name,
        #                                       data_dir=FLAGS.data_dir,
        #                                       max_seq_len=FLAGS.max_seq_len)
        raise NotImplementedError('PyTorch support coming soon to a sink near you!')
    else:
        raise NotImplementedError('Unsupported implementation \"%s\"' % FLAGS.implementation)

    # Load model
    model = experiment.load_model(model_name=FLAGS.model_name)

    # Train model
    logging.info('Training %s with %s...', FLAGS.model_name, ' '.join(FLAGS.tasks))
    experiment.train(model,
                     tasks=FLAGS.tasks,
                     num_epochs=FLAGS.num_epochs,
                     steps_per_epoch=FLAGS.steps_per_epoch,
                     prefetch_size=FLAGS.prefetch_size,
                     batch_size=FLAGS.batch_size,
                     eval_batch_size=FLAGS.eval_batch_size,
                     eval_batches=FLAGS.eval_batches,
                     checkpoint_file=FLAGS.checkpoint_file)

    # Evaluate the model
    logging.info('Evaluating %s with %s...', FLAGS.model_name, ' '.join(FLAGS.tasks))
    predictions = experiment.predict(model,
                                     tasks=FLAGS.tasks,
                                     eval_batch_size=FLAGS.eval_batch_size,
                                     eval_batches=FLAGS.eval_batches,
                                     splits=[tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST])

    logging.info('Results:')
    results = experiment.evaluate(predictions)
    print(results)


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
