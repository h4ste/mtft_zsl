import abc
from typing import TypeVar, Optional, Generic, Any, Text, Mapping, Sequence

import sklearn as skl
import tensorflow.keras as keras


class Dataset(abc.ABC):

    @property
    @abc.abstractmethod
    def x(self):
        pass

    @property
    @abc.abstractmethod
    def y(self):
        pass


Model = TypeVar('Model')


class Experiment(abc.ABC, Generic[Model]):

    @abc.abstractmethod
    def train(self, model: Model, train: Dataset, validation: Optional[Dataset] = None, class_weights=None):
        pass

    @abc.abstractmethod
    def evaluate(self, model: Model, test: Dataset) -> Mapping[Text, float]:
        pass

    @abc.abstractmethod
    def predict(self, model: Model, x: Any) -> Sequence[int]:
        pass


class TfExperiment(Experiment):

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def train(self, model: keras.Model, train: Dataset, validation: Optional[Dataset] = None, class_weights=None):
        model.fit(x=train.x,
                  y=train.y,
                  batch_size=self.batch_size,
                  validation_data=(validation.x, validation.y) if validation else None,
                  class_weights=class_weights
                  )

    def evaluate(self, model: keras.Model, test: Dataset) -> Mapping[Text, float]:
        raise NotImplementedError('Not yet implemented!')
        # model.evaluate(x=test.x,
        #                y=test.y,
        #                batch_size=self.batch_size)
        # return None

    def predict(self, model: keras.Model, x: Any) -> Sequence[float]:
        return model.predict(x=x,
                             batch_size=self.batch_size)


SklEstimator = TypeVar('SklEstimator', skl.tree.tree.BaseDecisionTree, skl.linear_model.base.LinearModel)


class SklExperiment(Experiment):

    def train(self, model: SklEstimator, train: Dataset, validation: None = None, class_weights=None):
        model.fit(X=train.x,
                  y=train.y)

    def evaluate(self, model: SklEstimator, test: Dataset) -> Mapping[Text, float]:
        raise NotImplementedError('Not yet implemented!')
        # y_pred = self.predict(x=test.x)
        # return None

    def predict(self, model: SklEstimator, x: Any) -> Sequence[float]:
        return model.predict(x=x)
