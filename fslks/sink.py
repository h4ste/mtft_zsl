import typing

import tensorflow as tf
import tensorflow_datasets.public_api as tfds
# noinspection PyProtectedMember
from tensorflow_datasets.core.registered import DatasetNotFoundError

import abc

from absl import logging

__SINK = {}

SEP = ' '


class Input(abc.ABC):

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        pass

    @abc.abstractmethod
    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        pass


class Constant(Input):
    def __init__(self, value: str):
        self._value = value

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return tf.constant(self._value, dtype=tf.string, name='sink.Constant')

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return str(self._value)


class Feature(Input):
    def __init__(self, key: str):
        self._key = key

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self._key in info.features

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return elem[self._key]

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return str(elem[self._key].numpy())


class DictEntry(Input):
    def __init__(self, dict_feature: str, entry_mapper: Input):
        self.dict_feature = dict_feature
        self.entry_mapper = entry_mapper

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self.dict_feature in info.features

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return self.entry_mapper.to_tensor(elem[self.dict_feature])

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return self.entry_mapper.to_str(elem[self.dict_feature])


class LabelMapping(Input):
    def __init__(self, label_feature: str, mapping: typing.Mapping[int, Input]):
        self.label = label_feature
        self.mapping = mapping

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self.label in info.features
        [input_.validate(info) for input_ in self.mapping.values()]

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return self.mapping[int(elem[self.label].numpy())].to_tensor(elem)

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return self.mapping[int(elem[self.label].numpy())].to_str(elem)


class Sequence(Input):
    def __init__(self, inputs: typing.Union[str, typing.Iterable[Input]]):
        self._inputs = inputs

    def validate(self, info: tfds.features.FeaturesDict) -> None:
        if isinstance(self._inputs, str):
            assert self._inputs in info.features
        elif isinstance(self._inputs, typing.Iterable):
            [input_.validate(info) for input_ in self._inputs]
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        if isinstance(self._inputs, str):
            inputs = Feature(self._inputs)
        elif isinstance(self._inputs, typing.Iterable):
            inputs = [input_.to_tensor(elem) for input_ in self._inputs]
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError

        return tf.strings.join(inputs, separator=' ', name='sink.Sequence')

    def to_str(self, elem: typing.Mapping[str, tf.Tensor]) -> str:
        if isinstance(self._inputs, str):
            return str(self._inputs)
        elif isinstance(self._inputs, typing.Iterable):
            return SEP.join(input_.to_str(elem) for input_ in self._inputs)
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError


def register(dataset_name: str, prompt: Input, input: Input, output: Input):
    try:
        builder = tfds.builder(dataset_name)
    except DatasetNotFoundError:
        logging.warning('Dataset %s was not found, it will not be registered to the kitchen sink.', dataset_name)
        return

    info = builder.info

    for text in [prompt, input, output]:
        text.validate(info)

    def make_conversion_fn(encoder_fn):

        def conversion_fn(elem: tfds.features.FeaturesDict):
            ex = encoder_fn(prompt.to_str(elem) + ' : ' + input.to_str(elem))
            outputs = encoder_fn(output.to_str(elem))['input_ids']
            return ex, outputs

        return conversion_fn

    __SINK[dataset_name] = make_conversion_fn


def get_converter(dataset_name: str):
    return __SINK[dataset_name]
