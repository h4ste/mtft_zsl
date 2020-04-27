import abc
import functools
import logging
import os
import typing

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import tqdm.auto as tqdm
import transformers

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

    data_dir: str = None

    def __init__(self, dataset: str, split: typing.Union[str, tfds.Split, None]):
        self.dataset = dataset
        self.split = split

    @classmethod
    def parse(cls, string: str):
        """Parses a command-line specified dataset and split string to determine the dataset and optionally the split
        e.g., "super_glue/copa" -> Task(dataset="super_glue/copa", split=None)
              "super_glue/copa::train" -> Task(dataset="super_glue/copa", split="train")
        :param string: dataset and split string,
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
            raise ValueError("Received unexpected dataset specification.")

        return Task(dataset, split)

    def _get_split_or_else(self, alternative: tfds.Split):
        if self.split is not None:
            return self.split
        elif Task.split_in_dataset(alternative, self.dataset):
            return alternative
        else:
            logging.warning('%s: dataset %s has no %s split and no alternative split was specified.',
                            self, self.dataset, alternative)
            return None

    @staticmethod
    def _parse_tasks(task_strs: typing.Iterable[str], fallback_split: typing.Optional[tfds.Split] = None):
        tasks = []
        for task_str in task_strs:
            task = Task.parse(task_str)

            if fallback_split:
                split = task._get_split_or_else(fallback_split)
                task = Task(dataset=task.dataset, split=split)

            if task.split is None:
                continue

            tasks.append(task)
        return tasks

    @staticmethod
    def parse_train_tasks(task_strs: typing.Iterable[str]):
        return Task._parse_tasks(task_strs, fallback_split=tfds.Split.TRAIN)

    @staticmethod
    def parse_validation_tasks(task_strs: typing.Iterable[str]):
        return Task._parse_tasks(task_strs, fallback_split=tfds.Split.VALIDATION)

    @staticmethod
    def parse_test_tasks(task_strs: typing.Iterable[str]):
        return Task._parse_tasks(task_strs, fallback_split=tfds.Split.TEST)

    def __str__(self):
        return '%s[%s]' % (self.dataset, self.split)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_or_load_dataset(name: str) -> (tfds.core.DatasetBuilder, tfds.core.DatasetInfo):
        builder: tfds.core.DatasetBuilder = tfds.builder(name, data_dir=Task.data_dir)
        builder.download_and_prepare()
        info: tfds.core.DatasetInfo = builder.info
        return builder, info

    @staticmethod
    def split_in_dataset(split: tfds.Split, dataset: str):
        _, info = Task.get_or_load_dataset(dataset)
        # logging.debug('Looking for %s in %s of %s', split, info.splits, dataset)
        return split in info.splits

    @classmethod
    def add_checksum_dir(cls, checksum_dir: str):
        if checksum_dir:
            tfds.download.add_checksums_dir(
                checksum_dir,
            )


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
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 seed: typing.Optional[int] = None):
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir

        if seed:
            np.random.seed(seed)
            # tf.random.set_seed(seed)
        self.tokenizer, self.encoder_fn, self.decoder_fn = self.load_tokenizer(tokenizer_name)

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
                                       truncation_strategy="only_first",
                                       return_token_type_ids=True,
                                       return_attention_mask=True)

        decoder_fn = functools.partial(tokenizer.decode, skip_special_tokens=True)

        return tokenizer, encoder_fn, decoder_fn

    def save_model(self, model: Model, path: str):
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @abc.abstractmethod
    def load_model(self, model_name: str) -> Model:
        pass

    def load_task_data(self, dataset: str, split: tfds.Split, decode=False) -> typing.Optional[tf.data.Dataset]:
        """ Loads the data for a given dataset

        :param dataset: Name of dataset to load
        :param split: Name of split to load
        :param decode: Whether to log & decode examples
        :return: a tf.data.Datasets
        """
        assert self.encoder_fn is not None
        assert self.decoder_fn is not None
        logging.debug('Loading %s data for dataset %s', split, dataset)
        builder, info = Task.get_or_load_dataset(dataset)
        data = builder.as_dataset(split=split, shuffle_files=True)

        # Load the converter for this dataset registered to the kitchen sink
        task_converter = sink.get_converter(dataset)(self.encoder_fn, self.decoder_fn if decode else None)

        logging.debug('Encoding %s[%s] to format required by Transformer...', dataset, split)
        tf_dataset = tf.data.Dataset.from_generator(
            lambda: map(task_converter, tqdm.tqdm(enumerate(data), desc='Tokenizing %s' % dataset, smoothing=1.)),
            output_types=(INPUT_TYPES, OUTPUT_TYPE, SAMPLE_WEIGHT_TYPE),
            output_shapes=({
                               "input_ids": tf.TensorShape([None]),
                               "attention_mask": tf.TensorShape([None]),
                               "token_type_ids": tf.TensorShape([None])
                           },
                           tf.TensorShape([None, 1]),
                           tf.TensorShape([None]))
        )
        if self.cache_dir == 'MEMORY':
            return tf_dataset.cache()
        elif self.cache_dir:
            cache_file = os.path.join(self.cache_dir, '%s.%s.cache' % (dataset, split))
            # If we have configuration details, they create intermediate directories that need to be created
            os.makedirs(os.path.join(cache_file, os.pardir), exist_ok=True)
            logging.debug('Caching tokenized data for %s[%s] to %s', dataset, split, cache_file)
            return tf_dataset.cache(cache_file)
        else:
            return tf_dataset

    def load_train_data(self,
                        tasks: typing.Sequence[Task],
                        batch_size: int,
                        prefetch_size: int) -> tf.data.Dataset:
        logging.debug('Loading training data...')
        training_data = []
        for task in tasks:
            dataset = self.load_task_data(task.dataset, task.split, decode=True) \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat()
            training_data.append(dataset)

        # This an array specifying which dataset should be used for each training iteration for one Epoch
        # Because tasks are already batched, we determine the number of batches in an epoch,
        # and sample a dataset for each batch.
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
        for task in tasks:
            task_data = self.load_task_data(task.dataset, task.split).batch(batch_size, drop_remainder=True)
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

    def _get_prediction_outputs(self,
                                model,
                                dataset,
                                split,
                                eval_batch_size: int,
                                eval_batches: typing.Optional[int] = None, ):
        decoder_fn = functools.partial(self.decoder_fn, clean_up_tokenization_spaces=True)

        task_data = self.load_task_data(dataset, split=split).batch(eval_batch_size, drop_remainder=False)

        if eval_batches:
            task_data = task_data.take(eval_batches)

        logging.info('Evaluating %s on %s', dataset, split)
        inputs = task_data.map(lambda inputs_, targets_, sample_weights: inputs_)

        outputs = self.predict_task_split(model, inputs)
        if outputs is None:
            logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                            dataset, split)
            return None

        targets = task_data.map(lambda inputs_, targets_, sample_weights: targets_).as_numpy_iterator()
        targets = np.concatenate(list(targets), axis=0)
        targets = np.squeeze(targets)

        input_ids = [ids for inputs_ in inputs.as_numpy_iterator() for ids in inputs_['input_ids']]
        for i, (inputs_, outputs_, targets_) in enumerate(zip(input_ids, outputs, targets), start=1):
            if i > LOG_EXAMPLES:
                break
            logging.info("Task %s[%s] Example %d Prompt: %s", dataset, split, i, decoder_fn(inputs_))
            logging.info("Task %s[%s] Example %d Outputs: %s", dataset, split, i, decoder_fn(outputs_))
            logging.info("Task %s[%s] Example %d Targets: %s", dataset, split, i, decoder_fn(targets_))

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

        for task in tasks:

            if task.dataset not in predictions:
                predictions[task.dataset]: TaskPredictions = {}

            predictions[task.dataset][task.split] = functools.partial(self._get_prediction_outputs,
                                                                      model=model,
                                                                      dataset=task.dataset,
                                                                      split=task.split,
                                                                      eval_batch_size=eval_batch_size,
                                                                      eval_batches=eval_batches)
        return predictions
