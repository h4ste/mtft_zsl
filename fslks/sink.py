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


class LabelError(Exception):

    def __init__(self, message: str = None):
        super().__init__()
        self._message = message

    def __str__(self):
        return self._message


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

    def __str__(self):
        return self._value


class Feature(Input):
    def __init__(self, key: str):
        self._key = key

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self._key in info.features, "\"%s\" was not a valid feature name!" % self._key

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return elem[self._key]

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return elem[self._key].decode('utf8')

    def __str__(self):
        return '[' + self._key + ']'


class DictEntry(Input):
    def __init__(self, dict_feature: str, entry_mapper: Input):
        self.dict_feature = dict_feature
        self.entry_mapper = entry_mapper

    def validate(self, info: tfds.core.DatasetInfo) -> None:
        assert self.dict_feature in info.features, "\"%s\" was not a valid feature name!" % self.dict_feature
        assert isinstance(info.features[self.dict_feature], tfds.features.FeaturesDict), \
            "\"%s\" was not a dictionary feature!" % self.dict_feature

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        return self.entry_mapper.to_tensor(elem[self.dict_feature])

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return self.entry_mapper.to_str(elem[self.dict_feature])

    def __str__(self):
        return '[' + self.dict_feature + '].' + str(self.entry_mapper)


class LabelMapping(Input):
    def __init__(self, label_feature: str, mapping: typing.Mapping[int, Input]):
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

    def to_str(self, elem: tfds.features.FeaturesDict) -> str:
        return self.mapping[int(elem[self.label])].to_str(elem)

    def __str__(self):
        return 'Mapping[%s]' % self.label


class Sequence(Input):
    def __init__(self, inputs: typing.Union[str, typing.Iterable[Input]]):
        self._inputs = inputs

    def validate(self, info: tfds.features.FeaturesDict) -> None:
        if isinstance(self._inputs, str):
            assert self._inputs in info.features, "\"%s\" was not a valid feature name!" % self._inputs
            assert isinstance(info.features[self._inputs], tfds.features.Sequence), \
                "\"%s\" was not a Sequence feature" % self._inputs
        elif isinstance(self._inputs, typing.Iterable):
            [input_.validate(info) for input_ in self._inputs]
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError

    def to_tensor(self, elem: tfds.features.FeaturesDict) -> tf.Tensor:
        if isinstance(self._inputs, str):
            inputs = Feature(self._inputs).to_tensor(elem)
        elif isinstance(self._inputs, typing.Iterable):
            inputs = [input_.to_tensor(elem) for input_ in self._inputs]
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError

        return tf.strings.join(inputs, separator=' ', name='sink.Sequence')

    def to_str(self, elem: typing.Mapping[str, tf.Tensor]) -> str:
        if isinstance(self._inputs, str):
            return SEP.join(input_.numpy().decode('utf8') for input_ in elem[self._inputs])
        elif isinstance(self._inputs, typing.Iterable):
            return SEP.join(input_.to_str(elem) for input_ in self._inputs)
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError

    def __str__(self):
        if isinstance(self._inputs, str):
            return '*[%s]' % self._inputs
        elif isinstance(self._inputs, typing.Iterable):
            return ' '.join(str(input_) for input_ in self._inputs)
        else:
            # This shouldn't happen if Python is correctly type checking!
            raise ValueError


def register(dataset_name: str, input: Input, target: Input, indicator: typing.Optional[Input] = None):
    try:
        builder = tfds.builder(dataset_name)
    except DatasetNotFoundError:
        logging.warning('Dataset %s was not found, it will not be registered to the kitchen sink.', dataset_name)
        return

    info = builder.info

    if indicator is not None:
        indicator.validate(info)
    input.validate(info)
    target.validate(info)
    logging.info('Registered %s with specification input:"<%s>" & targets: "<%s>"', dataset_name, input, target)

    def make_conversion_fn(encoder_fn, decoder_fn=None):

        def conversion_fn(idx: int, elem: tfds.features.FeaturesDict):
            segments = [indicator.to_tensor(elem)] if indicator else []
            segments.append(input.to_tensor(elem))
            input_ = tf.strings.join(segments, separator=' ')
            target_ = target.to_tensor(elem)
            return input_, target_

            # # idx, elem = idx_elem
            # try:
            #     segments = [indicator.to_str(elem)] if indicator else []
            #     segments.append(input.to_str(elem))
            #     input_ = SEP.join(segments)
            #     ex = encoder_fn(input_)
            # except TypeError as e:
            #     prompt_str = indicator.to_str(elem)
            #     input_str = input.to_str(elem)
            #     raise TypeError('In dataset %s:\nPrompt returned %s: %s\nInput returned %s: %s\n%s' % (
            #         dataset_name,
            #         type(prompt_str), prompt_str,
            #         type(input_str), input_str,
            #         e
            #     ))
            #
            # try:
            #     outputs = encoder_fn(target.to_str(elem))['input_ids']
            #     outputs = np.expand_dims(outputs, -1)
            # except tf.errors.UnknownError:
            #     raise LabelError()
            # if idx == 0 and decoder_fn is not None:
            #     logging.info('Task %s Example %d Input: %s', dataset_name, idx + 1, decoder_fn(ex['input_ids']))
            #     # logging.debug('Task %s Example %d Input Features: %s', dataset_name, idx + 1, ex)
            #     logging.info('Task %s Example %d Target: %s', dataset_name, idx + 1, decoder_fn(outputs))
            #
            # sample_weight = ex['attention_mask']
            # return ex, outputs, sample_weight

        return conversion_fn

    __SINK[dataset_name] = make_conversion_fn


def get_converter(dataset_name: str):
    return __SINK[dataset_name]
