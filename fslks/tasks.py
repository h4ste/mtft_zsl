import abc
import pathlib
from typing import Sequence, Union

import tensorflow as tf
import tensorflow.keras as keras

import tensorflow_datasets.public_api as tfds

import fslks as fsl

from fslks.modelling import QAModel

TASKS = {}


def register_task(name: Sequence[str], task: Task, tree=None):
    if not tree:
        tree = TASKS

    last = len(name) - 1
    for i, segment in enumerate(name):
        if segment not in tree:
            if i == last:
                tree[segment] = task
            else:
                tree[segment] = {}
                tree = tree[segment]


class Task(object, abc.ABC):

    def __init__(self,
                 name: Union[str, Sequence[str]],
                 dataset_builder: tfds.core.DatasetBuilder,
                 train_split: str = tfds.core.splits.Split.TRAIN,
                 validation_split: str = tfds.core.splits.Split.VALIDATION,
                 test_split: str = tfds.core.splits.Split.TEST,
                 loss: Union[str, keras.losses.Loss] = 'sparse_categorical_crossentropy',):
        if isinstance(name, str):
            name = name.split('/')
        register_task(name, self)

        self.name = name
        self.loss = loss
        self.builder = dataset_builder
        self.datasets = {}

    def prepare(self,
                batch_size=None,
                *args):
        self.builder.download_and_prepare()
        self.datasets = self.builder.as_dataset(batch_size=batch_size, *args)

    @property
    @abc.abstractmethod
    def train(self) -> fsl.Dataset:
        pass

    @property
    @abc.abstractmethod
    def validation(self) -> fsl.Dataset:
        pass

    @property
    @abc.abstractmethod
    def test(self) -> fsl.Dataset:
        pass

    @abc.abstractmethod
    def get_output(self, model: QAModel) -> Union[Sequence[tf.Tensor], tf.Tensor]:
        pass


class BinaryTask(Task):

    def __init__(self,
                 name: str,
                 train: str,
                 valid: str,
                 test: str,
                 n_classes: 2,
                 loss: Union[str, keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 folder: str = None, ):
        super().__init__(name=name, train=train, valid=valid, test=test, loss=loss, folder=folder)
        self.n_classes = n_classes

    def get_output(self, model: QAModel) -> tf.Tensor:
        x = model.outputs[0]
        x = keras.layers.Dense(self.n_classes)(x)
        return x


class LanguageGenerationTask(Task):
    def __init__(self,
                 name: str,
                 train: str,
                 valid: str,
                 test: str,
                 loss: Union[str, keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 folder: str = None):
        super().__init__(name=name, train=train, valid=valid, test=test, loss=loss, folder=folder)

    def get_output(self, model: QAModel) -> Sequence[tf.Tensor]:
        x = model.outputs
        x = model.vocab_decoder(x)
        x = keras.layers.Dense(model.get_vocabulary())(x)
        return x
