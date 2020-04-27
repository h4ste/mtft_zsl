import typing

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import transformers
from absl import logging

from fslks.experiments import Experiment
from fslks.experiments import Task


def configure_tf(use_xla: bool = False,
                 use_amp: bool = False) -> None:
    logging.info(('Enabling' if use_xla else 'Disabling') + ' XLA optimization')
    tf.config.optimizer.set_jit(use_xla)
    logging.info(('Enabling' if use_amp else 'Disabling') + ' auto mixed precision (AMP)')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_amp})


class TransformerOutputWrapper(keras.Model):

    def __init__(self, model: transformers.TFPreTrainedModel):
        super().__init__()
        self.model = model

    def call(self, inputs, **kwargs):
        outputs = self.model(inputs, **kwargs)
        if isinstance(outputs, tuple):
            logging.info('Outputs was a tuple, returning %s instead', outputs[0])
            return outputs[0]
        elif isinstance(outputs, tf.Tensor):
            return outputs
        else:
            raise ValueError('Unexpected outputs (type: %s): %s', type(outputs), outputs)


class TFExperiment(Experiment[tf.keras.Model]):

    def __init__(self,
                 tokenizer_name: str,
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 use_xla: bool = False,
                 use_amp: bool = True,
                 seed: typing.Optional[int] = None):
        super().__init__(tokenizer_name=tokenizer_name, max_seq_len=max_seq_len, cache_dir=cache_dir, seed=seed)
        configure_tf(use_xla=use_xla, use_amp=use_amp)

    def load_model(self, model_name: str) -> tf.keras.Model:
        model_name = model_name
        logging.info('Loading pre-trained TF model from %s', model_name)

        model: keras.Model
        if model_name.startswith('t5'):
            # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
            # like the others, so we need to load it separately.
            model = transformers.TFT5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = transformers.TFAutoModelWithLMHead.from_pretrained(model_name)

        return TransformerOutputWrapper(model)

    @staticmethod
    def compile_model(model: tf.keras.Model,
                      steps_per_epoch: int) -> None:
        lr = tfa.optimizers.Triangular2CyclicalLearningRate(
            initial_learning_rate=0.,
            maximal_learning_rate=1e-4,
            step_size=2 * steps_per_epoch,
        )
        opt = tfa.optimizers.LazyAdam(learning_rate=lr, epsilon=1e-08)

        if tf.config.optimizer.get_experimental_options().get('auto_mixed_precision'):
            logging.debug('Enabling loss scaling')
            opt = keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

        loss = keras.losses.SparseCategoricalCrossentropy(name='loss', reduction=keras.losses.Reduction.SUM)
        metric = keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer=opt, loss=loss, weighted_metrics=[metric], sample_weight_mode='temporal')

    def train(self,
              model: tf.keras.Model,
              training_tasks: typing.List[Task],
              validation_tasks: typing.List[Task],
              num_epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              prefetch_size: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> None:

        logging.info('Preparing kitchen sink with %d tasks: %s', len(training_tasks), training_tasks)
        TFExperiment.compile_model(model, steps_per_epoch)

        # Stop training if validation loss fails to decrease for 3 epochs
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                          patience=5,
                                          mode='max',
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]
        #
        # If requested, save model checkpoints
        if checkpoint_file:
            logging.info('Saving checkpoints to %s', checkpoint_file)
            callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                             monitor='val_accuracy',
                                                             save_best_only=True))

        # Train the model & return its training history
        logging.info('Beginning training...')
        training_data = self.load_train_data(training_tasks,
                                             batch_size=batch_size,
                                             prefetch_size=prefetch_size)

        validation_data = self.load_valid_data(validation_tasks,
                                               batch_size=eval_batch_size or batch_size,
                                               prefetch_size=prefetch_size,
                                               num_batches=eval_batches) if validation_tasks else None

        history = model.fit(x=training_data,
                            validation_data=validation_data,
                            epochs=num_epochs,
                            verbose=1,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)

        for metric, values in history.history.items():
            logging.info('%s: %s', metric, values)

    def predict_task_split(self, model, inputs: tf.data.Dataset) -> typing.Optional[np.ndarray]:
        try:
            logits = model.predict(inputs, verbose=1)
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

        logging.info('Logits Shape=%s; Logits=%s', logits.shape, logits)
        outputs = np.argmax(logits, axis=-1)
        return outputs
