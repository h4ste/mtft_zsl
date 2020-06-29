import abc
import functools
import logging
import os
import typing

import gorilla
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds
import transformers
from tensorflow.python import pywrap_tensorflow
# For memory leak:
from tensorflow.python.eager import context as tf_eager_context
from tensorflow.python.framework import random_seed

from fslks import sink
from fslks.sink import Task

# Type variable for Experiments
Model = typing.TypeVar('Model')

# Type aliases for Predictions table
TaskSplitPredictions = typing.MutableMapping[str, typing.Union[typing.Sequence[typing.Sequence[str]], np.ndarray]]
TaskPredictions = typing.MutableMapping[typing.Union[tfds.Split, str], typing.Callable[[], TaskSplitPredictions]]
Predictions = typing.MutableMapping[str, TaskPredictions]

LOG_EXAMPLES: int = 10


def concatenate(datasets: typing.Iterable[tf.data.Dataset]):
    dataset_itr = iter(datasets)

    # Start with the first dataset
    joint_dataset = next(dataset_itr)

    # Concatenate each remaining dataset
    for dataset in dataset_itr:
        joint_dataset = joint_dataset.concatenate(dataset)

    return joint_dataset


# Unsuccessful attempt at memory leak patch
# I didn't check to see how sample_from_dataset
# internally checks for the seed yet, but this fix
# was intended for tf.random.uniform originally
# which calls get_seed internally
_DEFAULT_OP_SEED = 1923746


class Experiment(abc.ABC, typing.Generic[Model]):
    def __init__(self,
                 configuration_name: str,
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 seed: typing.Optional[int] = None,
                 max_task_examples: float = 2e21,
                 temperature: float = 2.,
                 dynamic_mixing: bool = True):
        self.max_seq_len = max_seq_len
        self.cache_dir = cache_dir

        # Task mixing constants
        self.max_examples = max_task_examples
        self.temperature = temperature
        self.dynamic_mixing = dynamic_mixing

        if seed:
            logging.debug('Setting seed to %d', seed)
            self.seed = seed
            np.random.seed(seed)

            # Alternate get_seed, see https://github.com/lerobitaille/tf-issue-36164-workaround
            def _patched_get_seed(op_seed):
                if op_seed is not None:
                    return seed, op_seed
                else:
                    return seed, _DEFAULT_OP_SEED

            # Monkey batch get_seed from tf.random_seed
            patch_settings = gorilla.Settings(allow_hit=True, store_hit=True)
            seed_patch = gorilla.Patch(random_seed, 'get_seed', _patched_get_seed, settings=patch_settings)
            gorilla.apply(seed_patch)

        # Also clear the kernel cache, to reset any existing seeds
        _context = tf_eager_context.context()
        # noinspection PyProtectedMember
        if _context._context_handle is not None:
            # noinspection PyProtectedMember
            pywrap_tensorflow.TFE_ContextClearCaches(_context._context_handle)

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

    def load_task_data(self, dataset: str, split: tfds.Split, decode=False, train=False) -> \
            typing.Optional[tf.data.Dataset]:
        """ Loads the data for a given dataset

        :param dataset: Name of dataset to load
        :param split: Name of split to load
        :param decode: Whether to log & decode examples
        :param train: Whether we are training or not
        :return: a tf.data.Datasets
        """
        assert self.encoder_fn is not None
        assert self.decoder_fn is not None
        logging.debug('Loading %s data for dataset %s', split, dataset)
        builder, info = Task.get_or_load_dataset(dataset)
        # Get the size of the dataset
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

            max_seq_len = self.max_seq_len
            input_ids.set_shape([max_seq_len])
            attention_mask.set_shape([max_seq_len])
            token_type_ids.set_shape([max_seq_len]),
            target_ids.set_shape([max_seq_len])

            # Log first 5 inputs and targets for each dataset
            if idx < LOG_EXAMPLES and decode:
                tf.numpy_function(py_decode_and_log, [idx, 'Input', input_ids], [])
                tf.numpy_function(py_decode_and_log, [idx, 'Target', target_ids], [])

            # Prepare input dictionary
            input_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'task': dataset,
            }

            if self.config.is_encoder_decoder:
                if train:
                    start_ids = tf.constant(self.config.decoder_start_token_id, shape=[1], dtype=tf.int64)
                    decoder_ids = tf.concat([start_ids, target_ids[:-1]], axis=-1)
                    decoder_ids.set_shape([max_seq_len])
                    input_dict['decoder_input_ids'] = decoder_ids
                else:
                    input_dict['decoder_input_ids'] = input_ids

            # Add an extra dimension to targets to support temporal class weights in tf
            target_ids = tf.expand_dims(target_ids, axis=-1)

            return input_dict, target_ids, attention_mask

        logging.debug('Encoding %s[%s] to format required by Transformer...', dataset, split)
        return data.enumerate().map(convert)

    def maybe_cache(self, task: Task, data: tf.data.Dataset) -> tf.data.Dataset:
        if self.cache_dir == 'MEMORY':
            return data.cache()
        elif self.cache_dir:
            cache_file = os.path.join(self.cache_dir, self.config.model_type,
                                      '%s.%s.cache' % (task.dataset, task.split))
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

        logging.debug('Loading training data...')
        training_data = []
        dataset_sizes = []
        for task in tasks:
            dataset = self.load_task_data(task.dataset, task.split, decode=True, train=True)
            dataset = self.maybe_cache(task, dataset) \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat()
            training_data.append(dataset)

            # Find dataset size
            _, info = Task.get_or_load_dataset(task.dataset)
            dataset_sizes.append(info.splits[task.split].num_examples)

        training_tasks = [task.dataset for task in tasks]
        logging.debug('Mixing tasks with training sizes: %s',
                      dict(zip(training_tasks, map('{:,d}'.format, dataset_sizes))))

        # Dataset mixing if using more than one training dataset
        if len(dataset_sizes) > 1:
            # Take the sum of the size of the datasets, choosing between K and size of dataset n for each term
            # in the summation
            self.mixing_counts = tf.Variable(np.minimum(dataset_sizes, self.max_examples), dtype=tf.float32,
                                             trainable=False)
            self.mixing_rates = self.mixing_counts / tf.math.reduce_sum(self.mixing_counts)
            logging.debug('Proportional mixing rates: %s',
                          '; '.join(
                              '{:s}: {:0>5.2f}%'.format(t[0], t[1] * 100.)
                              for t in zip(training_tasks, self.mixing_rates.numpy()))
                          )
            smoothed_rates = tf.math.pow(self.mixing_rates, 1. / self.temperature)
            self.smoothed_mixing_rates = smoothed_rates / tf.math.reduce_sum(smoothed_rates)
            logging.debug('Smoothed mixing rates: %s',
                          '; '.join(
                              '{:s}: {:0>5.2f}%'.format(t[0], t[1] * 100.)
                              for t in zip(training_tasks, self.smoothed_mixing_rates.numpy()))
                          )
            # logging.info('Smoothed task rates: %s', dict(zip(tasks, map('{:5.2f}%'.format, smoothed_rates * 100.))))
            training_data = tf.data.experimental.sample_from_datasets(training_data,
                                                                      weights=self.smoothed_mixing_rates)
        else:
            training_data = training_data[0]

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
            else:
                task_data = task_data.cache()
            task_data = task_data.batch(batch_size, drop_remainder=True)
            if num_batches:
                task_data = task_data.take(num_batches).cache()
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
    def predict_task_split(self, model, data: tf.data.Dataset, task: Task, min_length: int, max_length: int) -> typing.Sequence[typing.Sequence[int]]:
        pass

    def _get_prediction_outputs(self,
                                model: Model,
                                task: Task,
                                eval_batch_size: int, ):
        
        decoder_fn = self.decoder_fn

        task_data = self.load_task_data(task.dataset, split=task.split)
        task_data = self.maybe_cache(task, task_data)
        task_data = task_data.batch(eval_batch_size, drop_remainder=False)

        targets = task_data.map(lambda inputs_, targets__, sample_weights: targets__).as_numpy_iterator()
        targets = np.concatenate(list(targets), axis=0)
        targets = np.squeeze(targets)

        target_tokens = [decoder_fn(targets_) for targets_ in targets]
        target_lens = [len(tokens.split()) for tokens in target_tokens]
        min_tokens = np.min(target_lens)
        max_tokens = np.max(target_lens)
        min_length = int(np.floor(min_tokens / 10) * 10)
        max_length = int(np.ceil(max_tokens / 10) * 10)
        logging.debug('Targets had minimum length %d (from %d) and maximum length %d (from %d)',
                      min_length, min_tokens, max_length, max_tokens)

        logging.info('Evaluating %s', task)
        inputs = task_data.map(lambda inputs_, targets__, sample_weights: inputs_)

        outputs = self.predict_task_split(model, inputs, task)
        if not outputs:
            logging.warning('Task %s has no labels for split %s, so it will not be evaluated.',
                            task.dataset, task.split)
            return None

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
                eval_batch_size: int) -> Predictions:
        predictions: Predictions = {}

        for task in tasks:

            if task.dataset not in predictions:
                predictions[task.dataset]: TaskPredictions = {}

            predictions[task.dataset][task.split] = functools.partial(self._get_prediction_outputs,
                                                                      model=model,
                                                                      task=task,
                                                                      eval_batch_size=eval_batch_size)
        return predictions
