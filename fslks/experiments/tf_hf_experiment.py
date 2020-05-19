import typing

import tensorflow.compat.v2 as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import transformers
from absl import logging
import tqdm.auto as tqdm

from fslks.experiments import Experiment
from fslks.experiments import Task


def configure_tf(use_xla: bool = False,
                 use_amp: bool = False) -> None:
    logging.info(('Enabling' if use_xla else 'Disabling') + ' XLA optimization')
    tf.config.optimizer.set_jit(use_xla)
    logging.info(('Enabling' if use_amp else 'Disabling') + ' auto mixed precision (AMP)')
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_amp})


class Logits2Softmax(keras.metrics.Metric):

    def __init__(self, metric: keras.metrics.Metric):
        super().__init__(name=f'l2s_{metric.name}', dtype=metric.dtype)
        self.metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.metric.update_state(y_true=y_true,
                                 y_pred=K.softmax(y_pred, axis=-1),
                                 sample_weight=sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_states()


class TransformerOutputWrapper(keras.Model):

    def __init__(self, model: transformers.TFPreTrainedModel):
        super().__init__()
        self.model = model
        self.save_pretrained = self.model.save_pretrained
        self.generate = self.model.generate

    def call(self, inputs, **kwargs):
        logging.debug('Attempting forward call with inputs: %s', inputs)
        outputs = self.model(inputs, **kwargs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if isinstance(outputs, tf.Tensor):
            return outputs
        else:
            raise ValueError('Unexpected outputs (type: %s): %s', type(outputs), outputs)


class T5OutputWrapper(TransformerOutputWrapper):

    def __init__(self, model: transformers.TFPreTrainedModel):
        assert isinstance(model, transformers.TFT5ForConditionalGeneration)
        super().__init__(model)

    def call(self, inputs, training=False, **kwargs):
        # T5 weirdly uses 'inputs' rather than input_ids. This was changed sometime after 2.8. No idea why.
        inputs['inputs'] = inputs['input_ids']
        del inputs['input_ids']

        # inputs['inputs'] = None

        return super().call(inputs=inputs, training=training, **kwargs)


class TFExperiment(Experiment[tf.keras.Model]):

    def __init__(self,
                 configuration_name: str,
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 use_xla: bool = False,
                 use_amp: bool = True,
                 seed: typing.Optional[int] = None):
        super().__init__(configuration_name=configuration_name, max_seq_len=max_seq_len, cache_dir=cache_dir, seed=seed)
        configure_tf(use_xla=use_xla, use_amp=use_amp)

    def load_model(self, model_name: str) -> tf.keras.Model:
        model_name = model_name
        logging.info('Loading pre-trained TF model from %s', model_name)

        model: keras.Model
        if model_name.startswith('t5'):
            # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
            # like the others, so we need to load it separately.
            model = transformers.TFT5ForConditionalGeneration.from_pretrained(model_name, config=self.config)
            return T5OutputWrapper(model)
        else:
            model = transformers.TFAutoModelWithLMHead.from_pretrained(model_name, config=self.config)
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

        loss = keras.losses.SparseCategoricalCrossentropy(name='loss',
                                                          reduction=keras.losses.Reduction.SUM,
                                                          from_logits=True)
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
                                          patience=3,
                                          mode='max',
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]

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

    def predict_task_split(self, model, inputs: tf.data.Dataset, task: Task):
        try:
            outputs = []
            for step, batch_inputs in enumerate(tqdm.tqdm(inputs)):
                logging.info('Batch Inputs: %s', batch_inputs)
                batch_outputs = model.generate(input_ids=batch_inputs['input_ids'],
                                               attention_mask=batch_inputs.get('attention_mask'),
                                               do_sample=True,
                                               max_length=140 + 2,
                                               min_length=55 + 1,
                                               num_beams=4,
                                               length_penalty=.6,
                                               no_repeat_ngram_size=3,
                                               early_stopping=True)
                outputs.extend(batch_outputs)
            return outputs
        # We can't just except tf.errors.UnknownError, because it is thrown as some sort of weird proxy
        # instance of a tf.errors.UnknownError and python's pattern matching can't handle the scandal
        except Exception as e:
            if isinstance(e, tf.errors.UnknownError):
                # Unfortunately, we don't get a more helpful error type, but this usually means
                # that the task has no labels for a given split (e.g., test evaluation occurs on a server)
                return []
            else:
                # We got a different exception type so let python freak out accordingly
                logging.warning('Encountered error: %s, %s', type(e), e)
                raise e
