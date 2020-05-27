import abc
import functools
import logging
import typing

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

__SINK = {}

SEP = ' '
PROMPT_END = ':'


def format_spaces(string):
    return '_'.join(string.split())


class LabelError(Exception):

    def __init__(self, message: str = None):
        super().__init__()
        self._message = message

    def __str__(self):
        return self._message


class Mapping(abc.ABC):

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        pass

    @abc.abstractmethod
    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        pass


class Constant(Mapping):
    def __init__(self, value: str):
        self._value = value

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return tf.constant(self._value, dtype=tf.string)

    def __str__(self):
        return self._value


class Feature(Mapping):
    def __init__(self, key: str):
        self._key = key

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self._key in info.features, "\"%s\" was not a valid feature name!" % self._key

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return elem[self._key]

    def __str__(self):
        return '[' + self._key + ']'


class DictEntry(Mapping):
    def __init__(self, dict_feature: str, entry_mapper: Mapping):
        self.dict_feature = dict_feature
        self.entry_mapper = entry_mapper

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self.dict_feature in info.features, "\"%s\" was not a valid feature name!" % self.dict_feature
        assert isinstance(info.features[self.dict_feature], tfds.features.FeaturesDict), \
            "\"%s\" was not a dictionary feature!" % self.dict_feature

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return tf.identity(self.entry_mapper.to_tensor(elem[self.dict_feature]))

    def __str__(self):
        return '[' + self.dict_feature + '].' + str(self.entry_mapper)


class LabelSwitch(Mapping):
    def __init__(self, label_feature: str, mapping: typing.Mapping[int, Mapping]):
        self.label = label_feature
        self.mapping = mapping

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self.label in info.features, "\"%s\" was not a valid feature name!" % self.label
        [input_.validate(info) for input_ in self.mapping.values()]

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        index = tf.dtypes.cast(elem[self.label], tf.int32)
        fns = {key: lambda input_=input_: input_.to_tensor(elem) for key, input_ in self.mapping.items()}
        tensor = tf.switch_case(branch_index=index,
                                branch_fns=fns,
                                default=lambda: 'NONE')
        return tensor

    def __str__(self):
        return 'Mapping[%s]' % self.label


class Join(Mapping):
    def __init__(self, inputs: typing.Iterable[Mapping], separator=' '):
        self._inputs = inputs
        self.separator = separator

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        [input_.validate(info) for input_ in self._inputs]

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        inputs = [input_.to_tensor(elem) for input_ in self._inputs]
        return tf.strings.join(inputs, separator=self.separator)

    def __str__(self):
        return ' '.join(str(input_) for input_ in self._inputs)


class Sequence(Mapping):
    def __init__(self, key: typing.Union[str, DictEntry]):
        self._key = key

    def validate(self, info: tfds.features.FeaturesDict) -> None:
        if isinstance(self._key, str):
            key = self._key
        elif isinstance(self._key, DictEntry):
            key = self._key.dict_feature
        else:
            raise ValueError('Unsupported key ' + str(type(self._key)) + ' for sink.Sequence')

        assert key in info.features, "\"%s\" was not a valid feature name!" % self._key
        assert isinstance(info.features[key], tfds.features.Sequence), \
            "\"%s\" was not a Sequence feature" % self._key

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        if isinstance(self._key, str):
            inputs = Feature(self._key).to_tensor(elem)
        elif isinstance(self._key, DictEntry):
            inputs = self._key.to_tensor(elem)
        else:
            raise ValueError('Unsupported key ' + str(type(self._key)) + ' for sink.Sequence')
        return tf.strings.reduce_join(inputs, separator=' ')

    def __str__(self):
        if isinstance(self._key, str):
            return '*[%s]' % self._key
        elif isinstance(self._key, DictEntry):
            return '*' + str(self._key)
        else:
            raise ValueError('Unsupported key ' + str(type(self._key)) + ' for sink.Sequence')


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
        builder.download_and_prepare(
            download_config=tfds.download.DownloadConfig(
                try_download_gcs=False
            )
        )
        info: tfds.core.DatasetInfo = builder.info
        return builder, info

    @staticmethod
    def split_in_dataset(split: typing.Union[str, tfds.Split], dataset: str):
        _, info = Task.get_or_load_dataset(dataset)
        # logging.debug('Looking for %s in %s of %s', split, info.splits, dataset)
        return split in info.splits

    @classmethod
    def add_checksum_dir(cls, checksum_dir: str):
        if checksum_dir:
            tfds.download.add_checksums_dir(
                checksum_dir,
            )


def register(dataset_name: str, input: Mapping, target: Mapping):
    try:
        _, info = Task.get_or_load_dataset(dataset_name)
        input.validate(info)
        target.validate(info)
    except IOError as ioe:
        logging.error('Unable to validate dataset %s', dataset_name, exc_info=ioe)

    logging.info('Registered %s with specification input:"<%s>" & targets: "<%s>"', dataset_name, input, target)

    def conversion_fn(elem: tfds.features.FeaturesDict):
        input_ = input.to_tensor(elem)
        target_ = target.to_tensor(elem)
        return input_, target_

    __SINK[dataset_name] = conversion_fn


def get_converter(dataset_name: str):
    return __SINK[dataset_name]
