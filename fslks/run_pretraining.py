import functools
import os
import typing

# Make TensorFlow print less obnoxious C logging messages
# (must be set before tensorflow is imported!)
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets.public_api as tfds
import tensorflow_addons as tfa

import transformers

from absl import flags
from absl import app
from absl import logging
from tabulate import tabulate

# We need to import our custom TensorFlow DataSet Builders
# noinspection PyUnresolvedReferences
from fslks import tasks
from fslks import sink
from fslks import eval

FLAGS = flags.FLAGS
flags.DEFINE_spaceseplist("tasks", None, "One or more tasks to be used for pretraining")

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
flags.DEFINE_string('checksum_dir', '/data/LHC_kitchensink/tensorflow_datasets/url_checksums',
                    help='Path to checksum directory')
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


def load_model(model_name: str) -> keras.Model:
    model_name = model_name
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


def concatenate(datasets: typing.Iterable[tf.data.Dataset]):
    dataset_itr = iter(datasets)

    # Start with the first dataset
    joint_dataset = next(dataset_itr)

    # Concatenate each remaining dataset
    for dataset in dataset_itr:
        joint_dataset = joint_dataset.concatenate(dataset)

    return joint_dataset


TQDM_BARS = {}


class Experiment(object):
    def __init__(self,
                 tokenizer_name: str,
                 data_dir: str,
                 max_seq_len: int,
                 prefetch_size: int = 10):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.prefetch_size = prefetch_size

        logging.debug('Loading tokenizer from %s...', tokenizer_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.encoder_fn = functools.partial(tokenizer.encode_plus,
                                            add_special_tokens=False,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            truncation_strategy="only_first")
        self.decoder_fn = functools.partial(tokenizer.decode, skip_special_tokens=True)

        tfds.download.add_checksums_dir(FLAGS.checksum_dir)

    def load_task_data(self, task: str, split: tfds.core.splits.NamedSplit, decode=False) -> (
            tf.data.Dataset, tf.data.Dataset):
        """ Loads the data for a given task

        :param task: Name of task to load
        :param split: Name of split to load
        :param decode: Whether to log & decode examples
        :return: a tf.data.Datasets
        """
        assert self.encoder_fn is not None
        assert self.decoder_fn is not None
        logging.debug('Loading %s data for task %s', split, task)
        data: tf.data.Dataset = tfds.load(task, split=split, data_dir=self.data_dir)

        # Load the converter for this task registered to the kitchen sink
        task_converter = sink.get_converter(task)(self.encoder_fn, self.decoder_fn if decode else None)

        logging.debug('Encoding %s[%s] (size = %s) to format required by Transformer...',
                      task, split, tf.data.experimental.cardinality(data).numpy())

        return tf.data.Dataset.from_generator(
            lambda: map(task_converter, enumerate(data)),
            output_types=(INPUT_TYPES, OUTPUT_TYPE),
            output_shapes=({
                               "input_ids": tf.TensorShape([None]),
                               "attention_mask": tf.TensorShape([None]),
                               "token_type_ids": tf.TensorShape([None])
                           }, tf.TensorShape([None]))
        )

    def load_train_data(self, tasks, batch_size):
        logging.debug('Loading training data...')
        training_data = []
        for task in tasks:
            dataset = self.load_task_data(task, tfds.Split.TRAIN, decode=True) \
                .cache() \
                .shuffle(128) \
                .batch(batch_size) \
                .repeat()
            training_data.append(dataset)

        # This an array specifying which task should be used for each training iteration for one Epoch
        # Because tasks are already batched, we determine the number of batches in an epoch,
        # and sample a task for each batch.
        choices = tf.data.Dataset.range(len(tasks)).repeat().shuffle(128)
        training_data = tf.data.experimental.choose_from_datasets(training_data, choices) \
            .prefetch(self.prefetch_size)
        return training_data

    def load_valid_data(self, tasks, batch_size):
        logging.debug('Loading validation data...')
        validation_data = []
        for task in tasks:
            try:
                task_data = self.load_task_data(task, tfds.Split.VALIDATION) \
                    .batch(batch_size)
            #                    .take(batch_size) \
            except ValueError as e:
                if str(e).startswith('Unknown split "validation"'):
                    # This is a ValueError indicating there is no validation split, so return nothing
                    logging.warning('Task %s has no validation split, so it will not be used for validation.', task)
                    continue
                else:
                    # This is some other ValueError and so should probably crash
                    raise e
            validation_data.append(task_data)

        return concatenate(validation_data)

    def train(self,
              model: keras.Model,
              tasks: typing.List[str],
              num_epochs: int,
              batch_size: int, eval_batch_size: int, samples_per_epoch: int,
              checkpoint_file: typing.Optional[str] = None) -> tf.keras.callbacks.History:
        logging.info('Preparing kitchen sink with %d tasks: %s', len(tasks), tasks)

        # Stop training if validation loss fails to decrease for 3 epochs
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   patience=5,
                                                   mode=max,
                                                   restore_best_weights=True)]

        # If requested, save model checkpoints
        if FLAGS.checkpoint_file:
            logging.info('Saving checkpoints to %s', checkpoint_file)
            callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                             monitor='val_accuracy',
                                                             save_best_only=True))
        # Train the model & return its training history
        logging.info('Beginning training...')
        history = model.fit(x=self.load_train_data(tasks, batch_size=batch_size),
                            validation_data=self.load_valid_data(tasks, batch_size=eval_batch_size),
                            epochs=num_epochs,
                            verbose=1,
                            steps_per_epoch=samples_per_epoch // batch_size,
                            callbacks=callbacks)

        return history

    def evaluate(self, model: tf.keras.Model, tasks: typing.List[str], eval_batch_size: int, eval_batches: int,
                 splits):
        headers = ['Task', 'Split', 'W. Acc.', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        results = []

        NA = 'n/a'
        for split in splits:
            for task in tasks:
                try:
                    task_data = self.load_task_data(task, split=split).batch(eval_batch_size)#.take(eval_batches)
                except ValueError as e:
                    if str(e).startswith('Unknown split'):
                        # This is a ValueError indicating there is no validation split, so return nothing
                        logging.warning('Task %s has no %s split, so it will not be evaluated.',
                                        task, split)
                        results.append([task, split] + [NA] * 5)
                        continue
                    else:
                        # This is some other ValueError so we should probably crash
                        raise e

                logging.info('Evaluating %s on %s', task, split)
                inputs = task_data.map(lambda inputs_, targets_: inputs_)
                try:
                    logits = model.predict(inputs, verbose=1)
                except tf.errors.UnknownError:
                    logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                                    task, split)
                    results.append([task, split] + [NA] * 5)
                    continue
                predictions = np.argmax(logits, axis=-1)
                logging.info("Task %s Split %s Example 1 Predictions: %s", task, split, self.decoder_fn(predictions[0]))

                targets = np.asarray(*tfds.as_numpy(task_data.map(lambda inputs_, targets_: targets_)))
                logging.info("Task %s Split %s Example 1 Targets: %s", task, split, self.decoder_fn(targets[0]))

                w_acc = eval.word_accuracy(targets, predictions)
                bleus = eval.bleu(targets, predictions)
                rouges = eval.rouge(targets, predictions)
                results.append([task, split,
                                w_acc,
                                bleus[0] * 100.,
                                rouges['rouge_1/f_score'] * 100.,
                                rouges['rouge_2/f_score'] * 100.,
                                rouges['rouge_l/f_score'] * 100.])

        return tabulate(results, headers=headers)


def main(argv):
    del argv  # Unused.

    logging.set_verbosity(logging.DEBUG)

    experiment = Experiment(tokenizer_name=FLAGS.model_name,
                            data_dir=FLAGS.data_dir,
                            max_seq_len=FLAGS.max_seq_len)

    # Configure TensorFlow settings
    configure_tf(use_amp=FLAGS.use_amp, use_xla=FLAGS.use_xla)

    # Load model
    model = load_model(model_name=FLAGS.model_name)

    # Train model
    history = experiment.train(model,
                               tasks=FLAGS.tasks,
                               num_epochs=FLAGS.num_epochs,
                               samples_per_epoch=FLAGS.samples_per_epoch,
                               batch_size=FLAGS.batch_size,
                               eval_batch_size=FLAGS.eval_batch_size,
                               checkpoint_file=FLAGS.checkpoint_file)
    # Print final results
    for metric, value in history.history.items():
        print(metric, '=', value)

    # Evaluate the model
    results = experiment.evaluate(model,
                                  tasks=FLAGS.tasks,
                                  eval_batch_size=FLAGS.eval_batch_size,
                                  eval_batches=1,
                                  splits=[tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST])

    print(results)


if __name__ == '__main__':
    # This is how abseil knows to parse arguments and flags
    app.run(main)
