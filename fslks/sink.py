import abc
import typing

import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import logging
# noinspection PyProtectedMember
from tensorflow_datasets.core.registered import DatasetNotFoundError

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
    def __init__(self, inputs: typing.Iterable[Mapping]):
        self._inputs = inputs

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        [input_.validate(info) for input_ in self._inputs]

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        inputs = [input_.to_tensor(elem) for input_ in self._inputs]
        return tf.strings.join(inputs, separator=' ')

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


def register(dataset_name: str, input: Mapping, target: Mapping):
    try:
        builder = tfds.builder(dataset_name)
    except DatasetNotFoundError:
        logging.warning('Dataset %s was not found, it will not be registered to the kitchen sink.', dataset_name)
        return

    info = builder.info

    input.validate(info)
    target.validate(info)
    logging.info('Registered %s with specification input:"<%s>" & targets: "<%s>"', dataset_name, input, target)

    def conversion_fn(elem: tfds.features.FeaturesDict):
        input_ = input.to_tensor(elem)
        target_ = target.to_tensor(elem)
        return input_, target_

    __SINK[dataset_name] = conversion_fn


def get_converter(dataset_name: str):
    return __SINK[dataset_name]
