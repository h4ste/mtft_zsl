import functools
import typing
import os

# Make TensorFlow print less obnoxious C logging messages
# (must be set before tensorflow is imported!)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets.public_api as tfds
import tensorflow_addons as tfa

import tqdm
import transformers

from absl import app
from absl import flags
from absl import logging

from fslks import sink
# We need to import our custom TensorFlow DataSet Builders
# noinspection PyUnresolvedReferences
from fslks import tasks as _

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("tasks", None, "One or more tasks to be used for pretraining")

flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 32, 'Batch size to use for training')
flags.DEFINE_integer('eval_batch_size', 64, 'Batch size to use when evaluating validation/test sets')
flags.DEFINE_boolean('use_xla', False, 'Enable XLA optimization')
flags.DEFINE_boolean('use_amp', False, 'Enable AMP optimization')
flags.DEFINE_integer('max_seq_len', 128, 'Maximum sequence length')
flags.DEFINE_string('model_name', 'bert-base-cased', 'Name of pretrained transformer model to load')
flags.DEFINE_string('checkpoint_file', None, 'Path to save checkpoints')
flags.DEFINE_string('data_dir', None, 'Path to TensorFlow DataSet home (e.g., ~/tensorflow_datasets)')
flags.DEFINE_string('cache_dir', None, 'Path to save TensorFlow DataSet cache files (e.g., /tmp)')
flags.DEFINE_integer('samples_per_epoch', 10_000, 'Number of samples to select to form each epoch')

# The types of inputs provided to the Transformer
INPUT_TYPES: typing.Mapping[str, tf.dtypes.DType] = {
    "input_ids": tf.int32,
    "attention_mask": tf.int32,
    "token_type_ids": tf.int32
}

# Type type of output expected from the Transformer
OUTPUT_TYPE: tf.dtypes.DType = tf.int32


def configure_tf(use_xla: bool = False,
                 use_amp: bool = False) -> None:
    logging.info(('Enabling' if use_xla else 'Disabling') + ' XLA optimization')
    tf.config.optimizer.set_jit(use_xla)
    logging.info(('Enabling' if use_amp else 'Disabling') + ' auto mixed precision (AMP)')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_amp})


def get_model(model_name: str) -> keras.Model:
    logging.info('Loading pre-trained TF model from %s', model_name)
    model = transformers.TFAutoModelWithLMHead.from_pretrained(model_name)

    opt = tfa.optimizers.LazyAdam(learning_rate=3e-5, epsilon=1e-08)

    if tf.config.optimizer.get_jit():
        logging.debug('Enabling loss scaling')
        opt = keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    return model


def load_all_data(tokenizer_name: str,
                  tasks: typing.Sequence[str],
                  max_seq_len: int,
                  batch_size: int,
                  eval_batch_size: int,
                  samples_per_epoch: int,
                  prefetch_size: int = 10,
                  data_dir: typing.Optional[str] = None,
                  cache_dir: typing.Optional[str] = None) -> (tf.data.Dataset, tf.data.Dataset):
    logging.debug('Loading tokenizer from %s...', tokenizer_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    def load_task_data(task: str, split: tfds.core.splits.NamedSplit) -> (tf.data.Dataset, tf.data.Dataset):
        """ Loads the data for a given task

        :param task: Name of task to load
        :param split: Name of split to load
        :return: a tuple containing training and validation tf.data.Datasets
        """
        logging.debug('Loading data for task %s', task)
        data: tf.data.Dataset = tfds.load(task, split=split, data_dir=data_dir)

        encoder_fn = functools.partial(tokenizer.encode_plus,
                                       add_special_tokens=False,
                                       max_length=max_seq_len,
                                       pad_to_max_length=True,
                                       return_attention_mask=True,
                                       truncation_strategy="only_first")

        # Load the converter for this task registered to the kitchen sink
        task_converter = sink.get_converter(task)(encoder_fn)

        logging.debug('Encoding %s[%s] (size = %s) to format required by Transformer...',
                      task, split, tf.data.experimental.cardinality(data).numpy())

        return tf.data.Dataset.from_generator(
            lambda: tqdm.tqdm(map(task_converter, data), desc='tokenizing'),
            output_types=(INPUT_TYPES, OUTPUT_TYPE),
            output_shapes=({
                               "input_ids": tf.TensorShape([None]),
                               "attention_mask": tf.TensorShape([None]),
                               "token_type_ids": tf.TensorShape([None])
                           }, tf.TensorShape([None]))
        )

    # This an array specifying which task should be used for each training iteration for one Epoch
    # Because tasks are already batched, we determine the number of batches in an epoch,
    # and sample a task for each batch.
    choices = tf.data.Dataset.range(len(tasks)).repeat(samples_per_epoch // batch_size).shuffle(128)

    def load_train_fn(task):
        dataset = load_task_data(task, tfds.Split.TRAIN).cache().shuffle(128).batch(batch_size).repeat()
        return dataset  # tfds.as_numpy(dataset)

    logging.debug('Loading training data...')
    training_data = list(map(load_train_fn, tasks))
    training_data = tf.data.experimental.choose_from_datasets(training_data, choices).prefetch(prefetch_size)

    def load_validation_fn(task):
        dataset = load_task_data(task, tfds.Split.VALIDATION).take(eval_batch_size).batch(eval_batch_size)
        return dataset  # tfds.as_numpy(dataset)

    logging.debug('Loading validation data...')
    validation_data = list(map(load_validation_fn, tasks))
    validation_prefetch = min(len(tasks), prefetch_size)
    validation_data = tf.data.experimental.choose_from_datasets(validation_data, choices).prefetch(validation_prefetch)

    return training_data, validation_data


def train(model: keras.Model,
          train_data: tf.data.Dataset,
          validation_data: tf.data.Dataset,
          num_epochs: int,
          checkpoint_file: typing.Optional[str] = None) -> tf.keras.callbacks.History:
    # Stop training if validation loss fails to decrease for 3 epochs
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                               patience=3,
                                               restore_best_weights=True),
                 ]

    # If requested, save model checkpoints
    if FLAGS.checkpoint_file:
        logging.info('Saving checkpoints to %s', checkpoint_file)
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                         monitor='val_accuracy',
                                                         save_best_only=True))
    # Train the model & return its training history
    logging.info('Beginning training...')
    return model.fit(x=train_data,
                     validation_data=validation_data,
                     epochs=num_epochs,
                     callbacks=callbacks)


def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    # Configure TensorFlow settings
    configure_tf(use_amp=FLAGS.use_amp, use_xla=FLAGS.use_xla)

    # Load the Kitchen Sink
    train_data, valid_data = load_all_data(tokenizer_name=FLAGS.model_name,
                                           tasks=FLAGS.tasks,
                                           max_seq_len=FLAGS.max_seq_len,
                                           batch_size=FLAGS.batch_size,
                                           eval_batch_size=FLAGS.eval_batch_size,
                                           data_dir=FLAGS.data_dir,
                                           cache_dir=FLAGS.cache_dir,
                                           samples_per_epoch=FLAGS.samples_per_epoch)

    # Create the transformer
    model = get_model(FLAGS.model_name)

    # Train the model
    history = train(model, train_data, valid_data,
                    num_epochs=FLAGS.num_epochs,
                    checkpoint_file=FLAGS.checkpoint_file)

    # Print final results
    for metric, value in history.history.items():
        print(metric, '=', value)


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
