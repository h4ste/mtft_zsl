import typing

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import torch
from poutyne.framework import Model

from fslks.experiments import Experiment
from absl import logging

import transformers


class TransformerOutputWrapper(poutyne.framework):

    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__()
        self.model = model

    def call(self, inputs, **kwargs):
        outputs = self.model(inputs, **kwargs)
        if isinstance(outputs, tuple):
            logging.info('Outputs was a tuple, returning %s instead', outputs[0])
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError('Unexpected outputs (type: %s): %s', type(outputs), outputs)


class PTExperiment(Experiment[torch.nn.Module]):

    def load_model(self, model_name: str) -> torch.nn.Module:
        model_name = model_name
        logging.info('Loading pre-trained PT model from %s', model_name)

        model: torch.nn.Module
        if model_name.startswith('t5'):
            # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
            # like the others, so we need to load it separately.
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = Model(transformers.AutoModelWithLMHead.from_pretrained(model_name), 'sgd', 'cross_entropy', batch_metrics=['accuracy'], epoch_metrics=['f1'])

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)

        return TransformerOutputWrapper(model)

    @staticmethod
    def compile_model(model: torch.nn.Module,
                      steps_per_epoch: int) -> None:
        pass
        #lr = tfa.optimizers.Triangular2CyclicalLearningRate(
        #    initial_learning_rate=0.,
        #    maximal_learning_rate=1e-4,
        #    step_size=2 * steps_per_epoch,
        #)
        #opt = tfa.optimizers.LazyAdam(learning_rate=lr, epsilon=1e-08)

        #if tf.config.optimizer.get_experimental_options().get('auto_mixed_precision'):
        #    logging.debug('Enabling loss scaling')
        #    opt = keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

        #loss = keras.losses.SparseCategoricalCrossentropy(name='loss', reduction=keras.losses.Reduction.SUM)
        #metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

        #model.compile(optimizer=opt, loss=loss, weighted_metrics=[metric], sample_weight_mode='temporal')

    def train(self,
              model: torch.nn.Module,
              tasks: typing.List[str],
              num_epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              prefetch_size: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> None:
        logging.info('Preparing kitchen sink with %d tasks: %s', len(tasks), tasks)

        PTExperiment.compile_model(model, steps_per_epoch)

        # Stop training if validation loss fails to decrease for 3 epochs
        #callbacks = [
        #    keras.callbacks.EarlyStopping(monitor='val_accuracy',
        #                                  patience=5,
        #                                  mode='max',
        #                                  restore_best_weights=True),
        #    keras.callbacks.TerminateOnNaN(),
        #]

        ## If requested, save model checkpoints
        #if checkpoint_file:
        #    logging.info('Saving checkpoints to %s', checkpoint_file)
        #    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
        #                                                     monitor='val_accuracy',
        #                                                     save_best_only=True))

        # Train the model & return its training history
        logging.info('Beginning training...')
        training_data = self.load_train_data(tasks,
                                             batch_size=batch_size,
                                             prefetch_size=prefetch_size)

        validation_data = self.load_valid_data(tasks,
                                               batch_size=eval_batch_size or batch_size,
                                               prefetch_size=prefetch_size,
                                               num_batches=eval_batches)

        training_data = tf.data.Dataset.as_numpy_iterator(training_data)
        validation_data = tf.data.Dataset.as_numpy_iterator(validation_data)
        history = model.fit(x=training_data,
                            validation_data=validation_data,
                            epochs=num_epochs,
                            verbose=1,
                            steps_per_epoch=steps_per_epoch,
                            )

        for metric, values in history.history.items():
            logging.info('%s: %s', metric, values)

    def predict_task_split(self, model, inputs: tf.data.Dataset) -> typing.Optional[np.ndarray]:
        try:
            logits = model.predict(inputs, verbose=1)
            logits = np.concat(logits, axis=0)
        # We can't just except tf.errors.UnknownError, because it is thrown as some sort of weird proxy
        # instance of a tf.errors.UnknownError and python's pattern matching can't handle the scandal
        except Exception as e:
            if isinstance(e, tf.errors.UnknownError):
                # Unfortunately, we don't get a more helpful error type, but this usually means
                # that the task has no labels for a given split (e.g., test evaluation occurs on a server)
                return None
            else:
                # We got a different exception type so let python freak out accordingly
                logging.warning('Encountered error: %s, %s', type(e), e)
                raise e
        return np.asarray(logits)



