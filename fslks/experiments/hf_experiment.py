import abc
import functools
import typing

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import transformers
from absl import logging

from fslks import sink

# Type variable for Experiments
Model = typing.TypeVar('Model')

# Type aliases for Predictions table
TaskSplitPredictions = typing.MutableMapping[str, typing.Union[typing.Sequence[typing.Sequence[str]], np.ndarray]]
TaskPredictions = typing.MutableMapping[typing.Union[tfds.Split, str], typing.Callable[[], TaskSplitPredictions]]
Predictions = typing.MutableMapping[str, TaskPredictions]

# The types of inputs provided to the Transformer
INPUT_TYPES: typing.Mapping[str, tf.dtypes.DType] = {
    "input_ids": tf.int32,
    "attention_mask": tf.int32,
    "token_type_ids": tf.int32
}

# Type type of target expected from the Transformer
OUTPUT_TYPE: tf.dtypes.DType = tf.int32

# Type of sample weights
SAMPLE_WEIGHT_TYPE: tf.dtypes.DType = tf.float32

LOG_EXAMPLES: int = 1


class Task(object):
    dataset: str
    split: typing.Union[str, tfds.Split, None]

    def __init__(self, dataset: str, split: typing.Union[str, tfds.Split, None]):
        self.dataset = dataset
        self.split = split

    @classmethod
    def parse(cls, string: str):
        """Parses a command-line specified task and split string to determine the dataset and optionally the split
        e.g., "super_glue/copa" -> Task(dataset="super_glue/copa", split=None)
              "super_glue/copa::train" -> Task(dataset="super_glue/copa", split="train")
        :param string: task and split string,
        :return: a new Task object
        """
        task = string.split("::")
        if len(task) == 1:
            dataset = task[0]
            split = None
        elif len(task) == 2:
            dataset = task[0]
            split = task[1]
        else:
            raise ValueError("Received unexpected task specification.")

        return Task(dataset, split)

    def __iter__(self):
        yield self.dataset
        yield self.split

    def __str__(self):
        return '%s[%s]' % (self.dataset, self.split)


def concatenate(datasets: typing.Iterable[tf.data.Dataset]):
    dataset_itr = iter(datasets)

    # Start with the first dataset
    joint_dataset = next(dataset_itr)

    # Concatenate each remaining dataset
    for dataset in dataset_itr:
        joint_dataset = joint_dataset.concatenate(dataset)

    return joint_dataset


class Experiment(abc.ABC, typing.Generic[Model]):
    def __init__(self,
                 tokenizer_name: str,
                 data_dir: str,
                 # checksum_dir: str,
                 max_seq_len: int):
        self.data_dir = data_dir
        # self.checksum_dir = checksum_dir
        self.max_seq_len = max_seq_len
        # self.prefetch_size = prefetch_size
        self.tokenizer, self.encoder_fn, self.decoder_fn = self.load_tokenizer(tokenizer_name)

    @functools.lru_cache(maxsize=None)
    def get_or_load_dataset(self, name: str) -> (tfds.core.DatasetBuilder, tfds.core.DatasetInfo):
        builder: tfds.core.DatasetBuilder = tfds.builder(name, data_dir=self.data_dir)
        info: tfds.core.DatasetInfo = builder.info
        return builder, info

    def split_in_dataset(self, split: tfds.Split, dataset: str):
        _, info = self.get_or_load_dataset(dataset)
        return split in info.splits

    def load_tokenizer(self, tokenizer_name: str):
        logging.debug('Loading tokenizer from %s...', tokenizer_name)
        tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

        if not tokenizer.pad_token:
            tokenizer.pad_token = '[PAD]'

        encoder_fn = functools.partial(tokenizer.encode_plus,
                                       add_special_tokens=False,
                                       add_space_before_punct_symbol=True,
                                       max_length=self.max_seq_len,
                                       pad_to_max_length=True,
                                       truncation_strategy="only_first")

        decoder_fn = functools.partial(tokenizer.decode, skip_special_tokens=True)

        return tokenizer, encoder_fn, decoder_fn

    def save_model(self, model: Model, path: str):
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @abc.abstractmethod
    def load_model(self, model_name: str) -> Model:
        pass

    def load_task_data(self, task: str, split: tfds.Split, decode=False) -> typing.Optional[tf.data.Dataset]:
        """ Loads the data for a given task

        :param task: Name of task to load
        :param split: Name of split to load
        :param decode: Whether to log & decode examples
        :return: a tf.data.Datasets
        """
        assert self.encoder_fn is not None
        assert self.decoder_fn is not None
        logging.debug('Loading %s data for task %s', split, task)
        builder, info = self.get_or_load_dataset(task)
        builder.download_and_prepare()
        data = builder.as_dataset(split=split)

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

    def load_train_data(self,
                        tasks: typing.Sequence[Task],
                        batch_size: int,
                        prefetch_size: int) -> tf.data.Dataset:
        logging.debug('Loading training data...')
        training_data = []
        for task, split in tasks:
            if not split:
                if not self.split_in_dataset(tfds.Split.TRAIN, task):
                    logging.warning(
                        'Task %s has no %s split and no alternative split was given, so it will not be used for training.',
                        task, tfds.Split.TRAIN)
                    continue
                else:
                    split = tfds.Split.TRAIN

            dataset = self.load_task_data(task, split, decode=True) \
                .cache() \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat()
            training_data.append(dataset)

        # This an array specifying which task should be used for each training iteration for one Epoch
        # Because tasks are already batched, we determine the number of batches in an epoch,
        # and sample a task for each batch.
        choices = tf.data.Dataset.range(len(tasks)).repeat().shuffle(128)
        training_data = tf.data.experimental.choose_from_datasets(training_data, choices)
        return training_data.prefetch(prefetch_size)

    def load_valid_data(self,
                        tasks: typing.Iterable[Task],
                        batch_size: int,
                        prefetch_size: int,
                        num_batches: typing.Optional[int] = None) -> tf.data.Dataset:
        logging.debug('Loading validation data...')
        validation_data = []
        for task, split in tasks:
            if not split:
                if not self.split_in_dataset(tfds.Split.VALIDATION, task):
                    logging.warning(
                        'Task %s has no %s split and no alternative split was given, so it will not be used for training.',
                        task, tfds.Split.VALIDATION)
                    continue
                else:
                    split = tfds.Split.VALIDATION

            task_data = self.load_task_data(task, split).batch(batch_size, drop_remainder=True)
            if num_batches:
                task_data = task_data.take(num_batches)
            validation_data.append(task_data)

        return concatenate(validation_data).prefetch(prefetch_size)

    @abc.abstractmethod
    def train(self,
              model: Model,
              training_tasks: typing.List[Task],
              validation_tasks: typing.List[Task],
              num_epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              prefetch_size: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> None:
        pass

    @abc.abstractmethod
    def predict_task_split(self, model, data: tf.data.Dataset) -> np.ndarray:
        pass

    def _get_prediction_outputs(self, model, task, split,
                                eval_batch_size: int,
                                eval_batches: typing.Optional[int] = None, ):
        decoder_fn = functools.partial(self.decoder_fn, clean_up_tokenization_spaces=True)

        # try:
        task_data = self.load_task_data(task, split=split).batch(eval_batch_size, drop_remainder=False)
        # except ValueError as e:
        #     if str(e).startswith('Unknown split'):
        #         # This is a ValueError indicating there is no validation split, so return nothing
        #         logging.warning('Task %s has no %s split, so it will not be evaluated.',
        #                         task, split)
        #         continue
        #     else:
        #         # This is some other ValueError so we should probably crash
        #         raise e

        if eval_batches:
            task_data = task_data.take(eval_batches)

        logging.info('Evaluating %s on %s', task, split)
        inputs = task_data.map(lambda inputs_, targets_, sample_weights: inputs_)

        outputs = self.predict_task_split(model, inputs)
        if outputs is None:
            logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                            task, split)
            return None

        targets = task_data.map(lambda inputs_, targets_, sample_weights: targets_).as_numpy_iterator()
        targets = np.concatenate(list(targets), axis=0)
        targets = np.squeeze(targets)

        input_ids = [ids for inputs_ in inputs.as_numpy_iterator() for ids in inputs_['input_ids']]
        for i, (inputs_, outputs_, targets_) in enumerate(zip(input_ids, outputs, targets), start=1):
            if i > LOG_EXAMPLES:
                break
            logging.info("Task %s[%s] Example %d Prompt: %s", task, split, i, decoder_fn(inputs_))
            logging.info("Task %s[%s] Example %d Outputs: %s", task, split, i, decoder_fn(outputs_))
            logging.info("Task %s[%s] Example %d Targets: %s", task, split, i, decoder_fn(targets_))

        return {
            'prompt': [decoder_fn(inputs_) for inputs_ in input_ids],
            'predictions': [decoder_fn(output_) for output_ in outputs],
            'targets': [decoder_fn(targets_) for targets_ in targets],
        }

    def predict(self,
                model: Model,
                tasks: typing.List[Task],
                eval_batch_size: int,
                eval_batches: typing.Optional[int] = None, ) -> Predictions:
        predictions: Predictions = {}

        for task, split in tasks:
            if not split:
                if not self.split_in_dataset(tfds.Split.TEST, task):
                    logging.warning(
                        'Task %s has no %s split and no alternative split was specified, so it will not be predicted.',
                        task, tfds.Split.TEST)
                    continue
                else:
                    split = tfds.Split.TEST

            if task not in predictions:
                predictions[task]: TaskPredictions = {}

            predictions[task][split] = functools.partial(self._get_prediction_outputs,
                                                         model=model,
                                                         task=task,
                                                         split=split,
                                                         eval_batch_size=eval_batch_size,
                                                         eval_batches=eval_batches)
        return predictions
