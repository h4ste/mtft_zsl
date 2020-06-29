import logging
import typing
from collections import Counter

import numpy as np
import tensorflow as tf
import torch
import torch.nn
import tqdm.auto as tqdm
import transformers

from fslks.experiments import Experiment, Task

try:
    from apex import amp

    _has_apex = True
except ImportError:
    amp = None
    _has_apex = False


def is_apex_available():
    return _has_apex


INPUT_DTYPES = {
    'input_ids': torch.long,
    'decoder_input_ids': torch.long,
    'attention_mask': torch.float32,
    'token_type_ids': torch.long,
    'position_ids': torch.long,
    'head_mask': torch.float32
}

GENERATION_MAX_LENGTHS = {
    'duc/2007': 300,
    'duc/2004': 40,
}

GENERATION_MIN_LENGTHS = {
    'duc/2007': 140,
    'duc/2004': 5,
}


def get_optimizer(model: torch.nn.Module,
                  num_training_steps: int,
                  num_warmup_steps: int,
                  weight_decay: float = 0.,
                  lr: float = 5e-5,
                  epsilon: float = 1e-8) -> typing.Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


class PTExperiment(Experiment[transformers.PreTrainedModel]):

    def __init__(self,
                 configuration_name: str,
                 max_seq_len: int,
                 cache_dir: typing.Optional[str] = None,
                 warmup_epochs: int = 3,
                 max_grad_norm: int = 1,
                 gradient_accumulation_steps: int = 1,
                 use_amp: bool = True,
                 seed: typing.Optional[int] = None,
                 temperature: float = 2.,
                 dynamic_mixing: bool = False,
                 mix_from_validation: bool = True,
                 clip_mixing_size: float = 2e19):
        tf.config.experimental.set_visible_devices([], 'GPU')
        super().__init__(configuration_name=configuration_name,
                         max_seq_len=max_seq_len,
                         cache_dir=cache_dir,
                         seed=seed,
                         temperature=temperature,
                         dynamic_mixing=dynamic_mixing,
                         max_task_examples=clip_mixing_size)
        self.warmup_epochs = warmup_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mix_from_validation = mix_from_validation
        if seed:
            torch.manual_seed(seed)

    def load_model(self, model_name: str) -> transformers.PreTrainedModel:
        model_name = model_name
        logging.info('Loading pre-trained PT model from %s', model_name)

        model: transformers.PreTrainedModel
        if 't5' in model_name:
            # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
            # like the others, so we need to load it separately.
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name, config=self.config)
        elif 'bart' in model_name:
            # Fix for BART config
            self.config.update({"output_past": True})
            model = transformers.BartForConditionalGeneration.from_pretrained(model_name, config=self.config)
        else:
            model = transformers.AutoModelWithLMHead.from_pretrained(model_name, config=self.config)

        return model

    def prepare_forward_inputs(self, model: transformers.PreTrainedModel,
                               inputs: typing.Mapping[str, np.ndarray],
                               labels: typing.Optional[np.ndarray] = None) -> typing.Mapping[str, torch.Tensor]:
        # Okay, so, each model in HuggingFace has an associated tokenizer. The tokenizer is supposed to
        # return only the inputs that the associated model wants. Unfortunately, it absolutely does *NOT*
        # do this. Moreover, it doesn't even return outputs in the correct types, expected by the models,
        # so we need to iterate through all the tensors returned by the tokenizer (i.e., inputs) and
        # (a) check if they are in the signature of the model's forward method (gross)
        # and then (b) cast them to the correct type and put them on the GPU
        forward_arg_names = set(model.forward.__code__.co_varnames)

        params = {k: torch.from_numpy(v).to(device=self.device, dtype=INPUT_DTYPES[k])
                  for k, v in inputs.items()
                  if k in forward_arg_names}

        if labels is not None:
            labels = torch.from_numpy(np.squeeze(labels)).to(device=self.device, dtype=torch.long)

            # We have to do similar gymnastics because the models like to arbitrary rename their "labels" parameter
            if 'labels' in forward_arg_names:
                params['labels'] = labels
            elif 'lm_labels' in forward_arg_names:
                params['lm_labels'] = labels
            else:
                raise ValueError('Model %s did not have argument labels or lm_labels, only: %s' % (
                    model, forward_arg_names))

        return params

    def _train_step(self,
                    model: torch.nn.Module,
                    inputs: typing.Dict[str, np.ndarray],
                    labels: np.ndarray,
                    optimizer: torch.optim.Optimizer) -> float:
        model.train()

        # Run the forward pass
        loss = model(**self.prepare_forward_inputs(model, inputs, labels))[0]

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        if self.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def get_mixing_rate(self, tasks, rates, normalize=True, temperature=None):
        temperature = temperature or self.temperature
        mixing_rates = np.maximum(np.minimum(rates, self.max_examples), 1e-8)
        if normalize:
            mixing_rates /= np.sum(mixing_rates)
            logging.debug('Proportional mixing rates: %s',
                          '; '.join(
                              '{:s}: {:0>5.2f}%'.format(t[0], t[1] * 100.)
                              for t in zip(tasks, mixing_rates)
                          ))
        smoothed_rates = mixing_rates ** (1. / temperature)
        smoothed_rates /= np.sum(smoothed_rates)
        logging.debug('Smoothed mixing rates: %s',
                      '; '.join(
                          '{:s}: {:0>5.2f}%'.format(t[0], t[1] * 100.)
                          for t in zip(tasks, smoothed_rates)))
        return smoothed_rates

    def load_train_data(self,
                        tasks: typing.Sequence[Task],
                        batch_size: int,
                        prefetch_size: int):

        logging.debug('Loading training data...')
        training_data = []
        dataset_sizes = []
        for task in tasks:
            dataset = self.load_task_data(task.dataset, task.split, decode=True, train=True)
            dataset = self.maybe_cache(task, dataset) \
                .shuffle(128) \
                .batch(batch_size, drop_remainder=True) \
                .repeat() \
                .prefetch(prefetch_size) \
                .as_numpy_iterator()
            training_data.append(dataset)
            _, info = Task.get_or_load_dataset(task.dataset)
            dataset_sizes.append(info.splits[task.split].num_examples)
            logging.debug('Loaded %d examples for %s', info.splits[task.split].num_examples, task)
        return training_data, np.array(dataset_sizes)

    def train(self,
              model: transformers.PreTrainedModel,
              training_tasks: typing.List[Task],
              validation_tasks: typing.List[Task],
              num_epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              prefetch_size: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> None:
        logging.info('Preparing kitchen sink with %d training tasks: %s', len(training_tasks), training_tasks)

        # Train the model & return its training history
        logging.info('Beginning training...')
        training_data, data_sizes = self.load_train_data(training_tasks,
                                                         batch_size=batch_size,
                                                         prefetch_size=prefetch_size)

        if validation_tasks:
            logging.info('Preparing kitchen sink with %d validation tasks: %s', len(validation_tasks), validation_tasks)
            validation_data = self.load_valid_data(validation_tasks,
                                                   batch_size=eval_batch_size or batch_size,
                                                   prefetch_size=prefetch_size,
                                                   num_batches=eval_batches)
        else:
            validation_data = None
            logging.info('Preparing kitchen sink without validation')

        num_epochs += self.warmup_epochs
        optimizer, scheduler = get_optimizer(model,
                                             num_warmup_steps=self.warmup_epochs * steps_per_epoch,
                                             num_training_steps=num_epochs * steps_per_epoch)

        model.to(self.device)
        if self.use_amp:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        global_step = 0
        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_itr = tqdm.trange(0, num_epochs * steps_per_epoch, desc="Training", unit="batch")
        tasks = [task.dataset for task in training_tasks]
        mixing_rates = self.get_mixing_rate(tasks, data_sizes)
        total_task_steps = Counter({task: np.float32(0.) for task in tasks})
        for epoch in range(1, num_epochs + 1):
            epoch_itr = tqdm.trange(0, steps_per_epoch, desc="Epoch %d" % epoch, leave=False, unit="batch")
            epoch_task_steps = Counter({task: np.float32(0.) for task in tasks})
            running_task_losses = {task: np.float32(0.) for task in tasks}
            for step in epoch_itr:
                inputs, labels, _ = next(np.random.choice(training_data, p=mixing_rates))
                step_loss = self._train_step(model, inputs, labels, optimizer)
                tr_loss += step_loss
                train_itr.update()
                task = inputs['task'][0].decode('UTF-8')
                epoch_task_steps[task] += 1
                running_task_losses[task] += step_loss

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        self.gradient_accumulation_steps >= steps_per_epoch == (step + 1)):
                    if self.use_amp:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

            total_tasks = sum(epoch_task_steps.values())

            print('Epoch %d: Empirical Mixing Rates: %s' % (
                epoch,
                '; '.join('{:s}: {:0>5.2f}%'.format(task, rate * 100. / total_tasks)
                          for task, rate in epoch_task_steps.items())
            ))

            print('Epoch %d: Expected Mixing Rates: %s' % (
                epoch,
                '; '.join('{:s}: {:0>5.2f}%'.format(task, rate * 100.)
                          for task, rate in zip(tasks, mixing_rates))
            ))

            mixing_losses = [loss / epoch_task_steps[task] for task, loss in running_task_losses.items()]
            print('Epoch %d: Training Losses: %s' % (
                epoch,
                '; '.join('{:s}: {:g}'.format(task, loss) for task, loss in zip(tasks, mixing_losses))
            ))

            if epoch > self.warmup_epochs:
                total_task_steps += epoch_task_steps
                exploration_ratios = np.array([total_task_steps.get(task, np.float32(0)) / size
                                               for task, size in zip(tasks, data_sizes)])
                print('Epoch %d: Exploration Ratios: %s' % (
                    epoch,
                    '; '.join('{:s}: {:0>5.2f}%'.format(task, ratio * 100.)
                              for task, ratio in zip(tasks, exploration_ratios))
                ))

                if not self.mix_from_validation:
                    avg_loss = np.nanmean(mixing_losses)
                    mixing_losses = [er * loss + (1. - er) * avg_loss
                                     for er, loss in zip(exploration_ratios, np.nan_to_num(mixing_losses))]

            valid_steps = 0
            running_valid_loss = 0.
            if validation_data:
                epoch_task_steps = {task: np.float32(0.) for task in tasks}
                running_task_losses = {task: np.float32(0.) for task in tasks}
                with torch.no_grad():
                    for step, (inputs, labels, _) in enumerate(validation_data.as_numpy_iterator(), 1):
                        model.eval()
                        # Run the forward pass
                        valid_step_loss = model(**self.prepare_forward_inputs(model, inputs, labels))[0].item()
                        running_valid_loss += valid_step_loss
                        valid_task = inputs['task'][0].decode('UTF-8')
                        if valid_task in tasks:
                            epoch_task_steps[valid_task] += 1
                            running_task_losses[valid_task] += valid_step_loss
                        valid_steps += 1

                avg_val_loss = running_valid_loss / valid_steps
                # Save checkpoint if validation loss decreases and checkpoint dir has been provided
                if checkpoint_file:
                    if epoch == 1:
                        best_val_loss = avg_val_loss
                        logging.info("Saving best model with initial validation loss {0})".format(best_val_loss))
                        self.save_model(model, "{0}_best".format(checkpoint_file))
                    else:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            logging.info(
                                "Saving new best model with validation loss {0} (epoch {1})".format(best_val_loss,
                                                                                                    epoch))
                            self.save_model(model, "{0}_best".format(checkpoint_file))

                print('Epoch {:d}: Validation Losses: {:s}'.format(
                    epoch,
                    '; '.join('{:s}: {:g}'.format(task, loss / epoch_task_steps[task])
                              for task, loss in running_task_losses.items())
                ))

                if self.mix_from_validation:
                    mixing_losses = [loss / epoch_task_steps[task] for task, loss in running_task_losses.items()]

            if epoch > self.warmup_epochs and self.dynamic_mixing:
                new_mixing_rates = self.get_mixing_rate(
                    tasks=tasks,
                    rates=mixing_losses,
                    normalize=False,
                    temperature=(1. / self.temperature)
                )
                print('Epoch {:d}: Updating Mixing Rate: {:s}'.format(
                    epoch,
                    '; '.join(
                        '{:s}: {:0>5.2f}%->{:0>5.2f}% (Δ={:0>5.2f})'.format(
                            task,
                            old_rate * 100.,
                            smooth_rate * 100.,
                            (smooth_rate-old_rate) * 100.)
                        for task, old_rate, smooth_rate in zip(tasks, mixing_rates, new_mixing_rates))
                ))
                mixing_rates = new_mixing_rates
                logging.debug('Mixing rates (shape=%s; |tasks|=%d): %s', mixing_rates.shape, len(tasks), mixing_rates)

            lr = scheduler.get_last_lr()[0]
            loss_scalar = (tr_loss - logging_loss) / steps_per_epoch
            logging_loss = tr_loss
            train_itr.write('Global step: %d, lr: %g, loss: %g, val_loss: %g' % (
                global_step,
                lr,
                loss_scalar,
                running_valid_loss / valid_steps if valid_steps > 0 else np.NaN))

            if not np.isfinite(loss_scalar):
                logging.info('Loss was NaN, ending training after %d epochs.', epoch)
                train_itr.close()
                return

        train_itr.close()

    def predict_task_split(self,
                           model: transformers.PreTrainedModel,
                           inputs: tf.data.Dataset,
                           task: Task,
                           max_length: int = 140,
                           min_length: int = 55) -> typing.Sequence[typing.Sequence[int]]:

        try:
            outputs = []
            model.to(self.device)
            for batch_inputs in tqdm.tqdm(inputs.as_numpy_iterator(),
                                          desc="Predicting %s" % task,
                                          unit="batch", leave=False):
                with torch.no_grad():
                    model.eval()
                    forward_params = self.prepare_forward_inputs(model, batch_inputs)
                    batch_outputs = model.generate(forward_params['input_ids'],
                                                   attention_mask=forward_params['attention_mask'],
                                                   do_sample=False,
                                                   max_length=GENERATION_MAX_LENGTHS.get(task.dataset, max_length) + 2,
                                                   min_length=GENERATION_MIN_LENGTHS.get(task.dataset, min_length) + 1,
                                                   num_beams=4,
                                                   length_penalty=2.,
                                                   no_repeat_ngram_size=3,
                                                   early_stopping=True)

                    batch_outputs = batch_outputs.detach().cpu().numpy()
                    outputs.extend(batch_outputs)
            return outputs
        # We can't just except tf.errors.UnknownError, because it is thrown as some sort of weird proxy
        # instance of a tf.errors.UnknownError and python's pattern matching can't handle the scandal
        except Exception as e:
            if isinstance(e, tf.errors.UnknownError):
                logging.warning('Encountered error: %s on %s: %s', type(e), task, e)
                # Unfortunately, we don't get a more helpful error type, but this usually means
                # that the dataset has no labels for a given split (e.g., test evaluation occurs on a server)
                return []
            else:
                # We got a different exception type so let python freak out accordingly
                logging.error('Encountered error: %s on %s: %s', type(e), task, e)
                raise e
