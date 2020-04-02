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
flags.DEFINE_integer('batch_size', 128, 'Batch size to use for training')
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
flags.DEFINE_integer('samples_per_epoch', 10_000, 'Number of samples to select to form each epoch')

# The types of inputs provided to the Transformer
INPUT_TYPES: typing.Mapping[str, tf.dtypes.DType] = {
    "input_ids": tf.int32,
    "attention_mask": tf.int32,
    "token_type_ids": tf.int32
}

# Type type of output expected from the Transformer
OUTPUT_TYPE: tf.dtypes.DType = tf.int32

# Type of sample weights
SAMPLE_WEIGHT_TYPE: tf.dtypes.DType = tf.float32

# Type aliases for Predictions table
TaskSplitPredictions = typing.MutableMapping[str, typing.Union[typing.Sequence[str], np.ndarray]]
TaskPredictions = typing.MutableMapping[tfds.Split, TaskSplitPredictions]
Predictions = typing.MutableMapping[str, TaskPredictions]


class TransformerWrapper(keras.Model):

    def __init__(self, model: transformers.TFPreTrainedModel):
        super().__init__()
        self.model = model

    def call(self, inputs, **kwargs):
        outputs = self.model(inputs, **kwargs)
        if isinstance(outputs, tuple):
            logging.info('Outputs was a tuple, returning %s instead', outputs[0])
            return outputs[0]
        elif isinstance(outputs, tf.Tensor):
            return outputs
        else:
            raise ValueError('Unexpected outputs (type: %s): %s', type(outputs), outputs)


def configure_tf(use_xla: bool = False,
                 use_amp: bool = False) -> None:
    logging.info(('Enabling' if use_xla else 'Disabling') + ' XLA optimization')
    tf.config.optimizer.set_jit(use_xla)
    logging.info(('Enabling' if use_amp else 'Disabling') + ' auto mixed precision (AMP)')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_amp})


def load_model(model_name: str) -> keras.Model:
    model_name = model_name
    logging.info('Loading pre-trained TF model from %s', model_name)

    model: keras.Model
    if model_name.startswith('t5'):
        # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
        # like the others, so we need to load it separately.
        model = transformers.TFT5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = transformers.TFAutoModelWithLMHead.from_pretrained(model_name)

    return TransformerWrapper(model)


def concatenate(datasets: typing.Iterable[tf.data.Dataset]):
    dataset_itr = iter(datasets)

    # Start with the first dataset
    joint_dataset = next(dataset_itr)

    # Concatenate each remaining dataset
    for dataset in dataset_itr:
        joint_dataset = joint_dataset.concatenate(dataset)

    return joint_dataset


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
        tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = '[PAD]'
        self.encoder_fn = functools.partial(tokenizer.encode_plus,
                                            add_special_tokens=False,
                                            add_space_before_punct_symbol=True,
                                            max_length=max_seq_len,
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            truncation_strategy="only_first")
        self.decoder_fn = functools.partial(tokenizer.decode, skip_special_tokens=True)

        tfds.download.add_checksums_dir(FLAGS.checksum_dir)

    def load_task_data(self, task: str, split: tfds.Split, decode=False) -> tf.data.Dataset:
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

        logging.debug('Encoding %s[%s] to format required by Transformer...', task, split)
        return tf.data.Dataset.from_generator(
            lambda: map(task_converter, enumerate(data)),
            output_types=(INPUT_TYPES, OUTPUT_TYPE, SAMPLE_WEIGHT_TYPE),
            output_shapes=({
                               "input_ids": tf.TensorShape([None]),
                               "attention_mask": tf.TensorShape([None]),
                               "token_type_ids": tf.TensorShape([None])
                           },
                           tf.TensorShape([None, 1]),
                           tf.TensorShape([None]))
        )

    def load_train_data(self, tasks, batch_size):
        logging.debug('Loading training data...')
        training_data = []
        for task in tasks:
            dataset = self.load_task_data(task, tfds.Split.TRAIN, decode=True) \
                .cache() \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat()
            training_data.append(dataset)

        # This an array specifying which task should be used for each training iteration for one Epoch
        # Because tasks are already batched, we determine the number of batches in an epoch,
        # and sample a task for each batch.
        choices = tf.data.Dataset.range(len(tasks)).repeat().shuffle(128)
        training_data = tf.data.experimental.choose_from_datasets(training_data, choices) \
            .prefetch(self.prefetch_size)
        return training_data

    def load_valid_data(self, tasks, batch_size, num_batches=None):
        logging.debug('Loading validation data...')
        validation_data = []
        for task in tasks:
            try:
                task_data = self.load_task_data(task, tfds.Split.VALIDATION).batch(batch_size, drop_remainder=True)
                if num_batches:
                    task_data = task_data.take(num_batches)
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
              batch_size: int,
              samples_per_epoch: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> keras.callbacks.History:
        logging.info('Preparing kitchen sink with %d tasks: %s', len(tasks), tasks)

        # Stop training if validation loss fails to decrease for 3 epochs
        callbacks = [
            # keras.callbacks.EarlyStopping(monitor='val_accuracy',
            #                               patience=5,
            #                               mode='max',
            #                               restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]

        # If requested, save model checkpoints
        if FLAGS.checkpoint_file:
            logging.info('Saving checkpoints to %s', checkpoint_file)
            callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                             monitor='val_accuracy',
                                                             save_best_only=True))

        steps_per_epoch = samples_per_epoch // batch_size

        lr = tfa.optimizers.Triangular2CyclicalLearningRate(
            initial_learning_rate=0.,
            maximal_learning_rate=1e-4,
            step_size=2 * steps_per_epoch,
        )
        opt = tfa.optimizers.LazyAdam(learning_rate=lr, epsilon=1e-08)

        if tf.config.optimizer.get_experimental_options().get('auto_mixed_precision'):
            logging.debug('Enabling loss scaling')
            opt = keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

        loss = keras.losses.SparseCategoricalCrossentropy(name='loss', reduction=keras.losses.Reduction.SUM)
        metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer=opt, loss=loss, weighted_metrics=[metric], sample_weight_mode='temporal')

        # Train the model & return its training history
        logging.info('Beginning training...')
        training_data = self.load_train_data(tasks, batch_size=batch_size)
        logging.info('data=%s', training_data)
        validation_data = self.load_valid_data(tasks,
                                               batch_size=eval_batch_size or batch_size,
                                               num_batches=eval_batches)

        logging.info('validation=%s', validation_data)
        history = model.fit(x=training_data,
                            validation_data=validation_data,
                            epochs=num_epochs,
                            verbose=1,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)

        return history

    def predict(self,
                model: keras.Model,
                tasks: typing.List[str],
                splits: typing.Iterable[tfds.Split],
                eval_batch_size: int,
                eval_batches: typing.Optional[int] = None) -> Predictions:
        predictions: Predictions = {}
        for task in tasks:
            for split in splits:
                try:
                    task_data = self.load_task_data(task, split=split).batch(eval_batch_size, drop_remainder=False)
                except ValueError as e:
                    if str(e).startswith('Unknown split'):
                        # This is a ValueError indicating there is no validation split, so return nothing
                        logging.warning('Task %s has no %s split, so it will not be evaluated.',
                                        task, split)
                        continue
                    else:
                        # This is some other ValueError so we should probably crash
                        raise e

                if eval_batches:
                    task_data = task_data.take(eval_batches)

                logging.info('Evaluating %s on %s', task, split)
                inputs = task_data.map(lambda inputs_, targets_, sample_weights: inputs_)

                try:
                    logits = model.predict(inputs, verbose=1)
                # We can't just except tf.errors.UnknownError, because it is thrown as some sort of weird proxy
                # instance of a tf.errors.UnknownError and python's pattern matching can't handle the scandal
                except Exception as e:
                    if isinstance(e, tf.errors.UnknownError):
                        # Unfortunately, we don't get a more helpful error type, but this usually means
                        # that the task has no labels for a given split (e.g., test evaluation occurs on a server)
                        logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                                        task, split)
                        continue
                    else:
                        # We got a different exception type so let python freak out accordingly
                        logging.warning('Encountered error: %s, %s', type(e), e)
                        raise e

                outputs = np.argmax(logits, axis=-1)
                logging.info('Logits Shape=%s; Logits=%s', logits.shape, logits)
                logging.info('Outputs Shape=%s; Outputs=%s', outputs.shape, outputs)
                logging.info("Task %s Split %s Example 1 Predictions: %s", task, split, self.decoder_fn(outputs[0]))

                targets = task_data.map(lambda inputs_, targets_, sample_weights: targets_).as_numpy_iterator()
                targets = np.concatenate(list(targets), axis=0)
                targets = np.squeeze(targets)
                logging.info("Task %s Split %s Example 1 Targets: %s", task, split, self.decoder_fn(targets[0]))

                if task not in predictions:
                    predictions[task]: TaskPredictions = {}

                predictions[task][split] = {
                    'pred_logits': logits,
                    'pred_token_ids': outputs,
                    'pred_tokens': list(map(self.decoder_fn, outputs)),
                    'target_token_ids': targets,
                    'target_tokens': list(map(self.decoder_fn, targets))
                }
        return predictions

    def evaluate(self, predictions: Predictions) -> str:
        headers = ['Task', 'Split', 'W. Acc.', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        results = []

        for task, task_predictions in predictions.items():
            for split, split_predictions in task_predictions.items():
                targets = split_predictions['target_tokens']
                predictions_ = split_predictions['pred_tokens']
                try:
                    logging.debug('Evaluating %s[%s] W. Acc...', task, split)
                    w_acc = eval.word_accuracy(targets, predictions_)
                    logging.debug('Evaluating %s[%s] BLEU...', task, split)
                    bleus = eval.bleu(targets, predictions_)
                    logging.debug('Evaluating %s[%s] ROUGE...', task, split)
                    rouges = eval.rouge(targets, predictions_)
                    results.append([task, split,
                                    w_acc,
                                    bleus[0] * 100.,
                                    rouges['rouge_1/f_score'] * 100.,
                                    rouges['rouge_2/f_score'] * 100.,
                                    rouges['rouge_l/f_score'] * 100.])
                except ZeroDivisionError as e:
                    logging.warning('Division by zero when evaluating %s[%s]', task, split)
                    results.append([task, split, 0., 0., 0., 0., 0.])

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
    logging.info('Training %s with %s...', FLAGS.model_name, ' '.join(FLAGS.tasks))
    history = experiment.train(model,
                               tasks=FLAGS.tasks,
                               num_epochs=FLAGS.num_epochs,
                               samples_per_epoch=FLAGS.samples_per_epoch,
                               batch_size=FLAGS.batch_size,
                               eval_batch_size=FLAGS.eval_batch_size,
                               eval_batches=FLAGS.eval_batches,
                               checkpoint_file=FLAGS.checkpoint_file)

    # Print final results
    for metric, value in history.history.items():
        print(metric, '=', value)

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
