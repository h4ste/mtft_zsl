import abc
import functools
import logging
import os
import typing

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import transformers

from fslks import sink

# For memory leak:
from tensorflow.python.eager import context
from tensorflow.python.framework import random_seed
import gorilla
import random

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

    def __repr__(self):
        return self.__str__()

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
                 configuration_name: str,
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 seed: typing.Optional[int] = None):
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir
        # For dataset sizes
        self.dataset_info = {}

        if seed:
            np.random.seed(seed)
            # tf.random.set_seed(seed)

        logging.debug('Loading configuration from %s...')
        self.config: transformers.PretrainedConfig = transformers.AutoConfig.from_pretrained(configuration_name)

        logging.debug('Loading tokenizer from %s...', configuration_name)
        self.tokenizer: transformers.PreTrainedTokenizer = \
            transformers.AutoTokenizer.from_pretrained(configuration_name, config=self.config)

        # if not self.tokenizer.pad_token:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.config.pad_token_id = self.tokenizer.pad_token_id
        #     logging.warning('Tokenizer does not provide a pad token, using %s (id: %d)',
        #                     self.tokenizer.pad_token, self.tokenizer.pad_token_id)

        self.encoder_fn = functools.partial(self.tokenizer.encode_plus,
                                            add_special_tokens=False,
                                            add_space_before_punct_symbol=True,
                                            max_length=self.max_seq_len,
                                            pad_to_max_length=True,
                                            truncation_strategy="only_first",
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        self.decoder_fn = functools.partial(self.tokenizer.decode,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

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
        # Get the size of the dataset
        self.dataset_info[dataset] = info.splits[split].num_examples
        data: tf.data.Dataset = builder.as_dataset(split=split, shuffle_files=True)

        # tf.numpy_function can't handle dicts, so we need to flatten the output into a list
        def py_tokenize_example(string):
            try:
                string = string.decode('utf-8')
            except AttributeError as e:
                logging.exception('In %s[%s]: failed to decode %s', dataset, split, string, exc_info=e)
            ex = self.encoder_fn(string)
            return [ex['input_ids'], ex['attention_mask'], ex['token_type_ids']]

        # decode and log a set of token_ids
        def py_decode_and_log(idx, name, token_ids):
            tokens = self.decoder_fn(token_ids)
            logging.info('Task %s[%s] Example %d %s: %s', dataset, split, idx + 1, name.decode('utf-8'), tokens)

        # Load the converter for this dataset registered to the kitchen sink
        task_converter = sink.get_converter(dataset)

        # convert tfds features into those required by transformers
        def convert(idx, ex):
            # Create input and target feature sequences
            input_, target_ = task_converter(ex)

            # Tokenize inputs & targets
            output_types = [tf.int64, tf.int64, tf.int64]
            input_ids, attention_mask, token_type_ids = tf.numpy_function(py_tokenize_example, [input_], output_types)
            target_ids, _, _ = tf.numpy_function(py_tokenize_example, [target_], output_types)

            # Log first 5 inputs and targets for each dataset
            if idx < 5 and decode:
                tf.numpy_function(py_decode_and_log, [idx, 'Input', input_ids], [])
                tf.numpy_function(py_decode_and_log, [idx, 'Target', target_ids], [])

            # Prepare input dictionary
            input_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }

            # Add an extra dimension to targets to support temporal class weights in tf
            target_ids = tf.expand_dims(target_ids, axis=-1)

            return input_dict, target_ids, attention_mask

        logging.debug('Encoding %s[%s] to format required by Transformer...', dataset, split)
        return data.enumerate().map(convert)

    def maybe_cache(self, task: Task, data: tf.data.Dataset) -> tf.data.Dataset:
        if self.cache_dir == 'MEMORY':
            return data.cache()
        elif self.cache_dir:
            cache_file = os.path.join(self.cache_dir, self.config.model_type, '%s.%s.cache' % (task.dataset, task.split))
            # If we have configuration details, they create intermediate directories that need to be created
            os.makedirs(os.path.join(cache_file, os.pardir), exist_ok=True)
            logging.debug('Caching tokenized data for %s to %s', task, cache_file)
            return data.cache(cache_file)
        else:
            return data

    def load_train_data(self,
                        tasks: typing.Sequence[Task],
                        batch_size: int,
                        prefetch_size: int) -> tf.data.Dataset:

        # Unsucessful attempt at memory leak patch
        # I didn't check to see how sample_from_dataset
        # internally checks for the seed yet, but this fix 
        # was intended for tf.random.uniform originally
        # which calls get_seed internally
        DEFAULT_OP_SEED = 1923746
        # Defines the function
        def better_get_seed(global_seed, op_seed):
            if op_seed is not None:
                return global_seed, op_seed
            else:
                return global_seed, DEFAULT_OP_SEED

        # Monkey Patch get_seed.
        def set_seed(seed=100):
            np.random.seed(seed)
            # Monkey Patch get_seed.
            func = lambda op_seed: better_get_seed(seed, op_seed)
            settings = gorilla.Settings(allow_hit=True, store_hit=True)
            patch = gorilla.Patch(
                random_seed, 'get_seed', func, settings=settings)
            gorilla.apply(patch)
        
        set_seed()

        logging.debug('Loading training data...')
        training_data = []
        dataset_sizes = []
        for task in tasks:
            dataset = self.load_task_data(task.dataset, task.split, decode=True)
            dataset_sizes.append(self.dataset_info[task.dataset])
            dataset = self.maybe_cache(task, dataset) \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat()
            training_data.append(dataset)

        # Dataset mixing if using more than one training dataset
        if len(dataset_sizes) > 1:
            mixing_rates = []
            # Artifically large k 
            K = 2e21
            # Temperature for scaling. As T nears one, scaling becomes equal to proportional mixing with max K
            T = 2
            # Take the sum of the size of the datasets, choosing between K and size of dataset n for each term
            # in the summation
            min_summ = sum(map(lambda e: min(e, K), dataset_sizes))
            mixing_rates = [min(e, K) / min_summ for e in dataset_sizes]
            scaled_rates = [r**(1 / T) for r in mixing_rates]
            normalized_rates = [r / sum(scaled_rates) for r in scaled_rates] 
            logging.info("Normalized mixing rates: {}".format(normalized_rates))
            training_data = tf.data.experimental.sample_from_datasets(
                        training_data, weights=normalized_rates, seed=None
                        )
        else: 
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
            task_data = self.load_task_data(task.dataset, task.split)
            if not num_batches:
                task_data = self.maybe_cache(task, task_data)
            task_data = task_data.batch(batch_size, drop_remainder=True)
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
    def predict_task_split(self, model, data: tf.data.Dataset, task: Task) -> typing.Sequence[typing.Sequence[int]]:
        pass

    def _get_prediction_outputs(self,
                                model: Model,
                                task: Task,
                                eval_batch_size: int,
                                eval_batches: typing.Optional[int] = None, ):
        decoder_fn = self.decoder_fn

        task_data = self.load_task_data(task.dataset, split=task.split)
        if not eval_batches:
            task_data = self.maybe_cache(task, task_data)
        task_data = task_data.batch(eval_batch_size, drop_remainder=False)
        if eval_batches:
            task_data = task_data.take(eval_batches)

        logging.info('Evaluating %s', task)
        inputs = task_data.map(lambda inputs_, targets__, sample_weights: inputs_)

        outputs = self.predict_task_split(model, inputs, task)
        if not outputs:
            logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                            task.dataset, task.split)
            return None

        targets = task_data.map(lambda inputs_, targets__, sample_weights: targets__).as_numpy_iterator()
        targets = np.concatenate(list(targets), axis=0)
        targets = np.squeeze(targets)

        input_ids = [ids for inputs_ in inputs.as_numpy_iterator() for ids in inputs_['input_ids']]
        for i, (inputs_, outputs_, targets_) in enumerate(zip(input_ids, outputs, targets), start=1):
            if i > LOG_EXAMPLES:
                break
            logging.info("Task %s Example %d Prompt: %s", task, i, decoder_fn(inputs_))
            logging.info("Task %s Example %d Outputs: %s", task, i, decoder_fn(outputs_))
            logging.info("Task %s Example %d Targets: %s", task, i, decoder_fn(targets_))

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
                                                                      task=task,
                                                                      eval_batch_size=eval_batch_size,
                                                                      eval_batches=eval_batches)
        return predictions
