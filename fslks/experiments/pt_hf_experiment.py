import typing

import numpy as np
import tensorflow as tf
import torch
import torch.nn
from torch.optim.optimizer import Optimizer

import tqdm.auto as tqdm

import transformers
from absl import logging

from fslks.experiments import Experiment, Task

INPUT_DTYPES = {
    'input_ids': torch.long,
    'attention_mask': torch.float32,
    'token_type_ids': torch.long,
    'position_ids': torch.long,
    'head_mask': torch.float32
}


def get_optimizer(model: torch.nn.Module,
                  weight_decay: float = 0.,
                  lr: float = 5e-5,
                  epsilon: float = 1e-8) -> Optimizer:
    # Prepare optimizer for weight decay
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
    return optimizer


class PTExperiment(Experiment[transformers.PreTrainedModel]):

    def __init__(self,
                 tokenizer_name: str,
                 data_dir: str,
                 # checksum_dir: str,
                 max_seq_len: int,
                 warmup_epochs: int = 3,
                 max_grad_norm: int = 1,
                 gradient_accumulation_steps: int = 1,
                 use_amp: bool = True):
        super().__init__(tokenizer_name=tokenizer_name, data_dir=data_dir, max_seq_len=max_seq_len)
        self.warmup_epochs = warmup_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name: str) -> transformers.PreTrainedModel:
        model_name = model_name
        logging.info('Loading pre-trained PT model from %s', model_name)

        model: transformers.PreTrainedModel
        if model_name.startswith('t5'):
            # HuggingFace named T5's sequence generator "ConditionalGeneration" rather than "LanguageModeling"
            # like the others, so we need to load it separately.
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            model = transformers.AutoModelWithLMHead.from_pretrained(model_name)

        return model

    def get_forward_params(self, model,
                           inputs: typing.Mapping[str, np.ndarray],
                           labels: typing.Optional[np.ndarray] = None) -> typing.Mapping[str, torch.Tensor]:
        # Okay, so, each model in HuggingFace has an associated tokenizer. The tokenizer is supposed to
        # return only the inputs that the associated model wants. Unfortunately, it absolutely does *NOT*
        # do this. Moreover, it doesn't even return outputs in the correct types, expected by the models,
        # so we need to iterate through all the tensors returned by the tokenizer (i.e., inputs) and
        # (a) check if they are in the signature of the model's forward method (gross)
        # and then (b) cast them to the correct type and put them on the GPU
        forward_arg_names = model.forward.__code__.co_varnames

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
                raise ValueError('Forward method of %s did not have argument labels or lm_labels, only: %s' % (
                    model, forward_arg_names))
        else:
            if isinstance(model, transformers.T5ForConditionalGeneration):
                # We are dealing with T5 so we need to rename inputs
                params['decoder_input_ids'] = params['input_ids']
                params['decoder_attention_mask'] = params['attention_mask']

        return params

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
        training_data = self.load_train_data(training_tasks,
                                             batch_size=batch_size,
                                             prefetch_size=prefetch_size).as_numpy_iterator()

        logging.info('Preparing kitchen sink with %d validation tasks: %s', len(validation_tasks), validation_tasks)
        validation_data = self.load_valid_data(validation_tasks,
                                               batch_size=eval_batch_size or batch_size,
                                               prefetch_size=prefetch_size,
                                               num_batches=eval_batches)

        opt: Optimizer = get_optimizer(model)

        if self.use_amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model: transformers.PreTrainedModel
            model, opt = amp.initialize(model, opt, opt_level='O1')

        num_epochs += self.warmup_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=self.warmup_epochs * steps_per_epoch,
            num_training_steps=num_epochs * steps_per_epoch
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        model.to(self.device)
        model.zero_grad()
        global_step = 0
        epoch_itr = tqdm.trange(0, num_epochs * steps_per_epoch, desc="Training", )
        for epoch in range(1, num_epochs + 1):
            running_loss = 0.
            training_itr = tqdm.tqdm(training_data, desc="Epoch %d" % epoch, initial=1, leave=True, unit=" steps")
            training_steps = 0
            for step, (inputs, labels, _) in enumerate(training_itr, 1):
                epoch_itr.update()
                model.train()

                # Run the forward pass
                loss = model(**self.get_forward_params(model, inputs, labels))[0]

                # Run the backwards pass
                if self.use_amp:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                running_loss += loss.item()
                if step % self.gradient_accumulation_steps == 0:
                    params = amp.master_params(opt) if self.use_amp else model.parameters()
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                    opt.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                training_steps += 1
                if step == steps_per_epoch:
                    break

            valid_steps = 0
            running_valid_loss = 0.
            for step, (inputs, labels, _) in enumerate(validation_data.as_numpy_iterator(), 1):
                model.eval()
                with torch.no_grad():
                    # Run the forward pass
                    loss = model(**self.get_forward_params(model, inputs, labels))[0]
                    running_valid_loss += loss.item()
                valid_steps += 1

            training_itr.set_postfix_str('Global step: %d, tr_loss: %f, val_loss: %f' % (
                global_step,
                running_loss / training_steps,
                running_valid_loss / valid_steps if valid_steps > 0 else 'NaN'))

            training_itr.close()
        epoch_itr.close()

    def predict_task_split(self, model: transformers.PreTrainedModel, inputs: tf.data.Dataset) -> \
            typing.Optional[np.ndarray]:
        try:
            outputs = []
            model.to(self.device)
            for batch_inputs in inputs.as_numpy_iterator():
                with torch.no_grad():
                    model.eval()
                    # logging.debug('Batch inputs: %s', batch_inputs)
                    forward_params = self.get_forward_params(model, batch_inputs)
                    # logging.debug('Forward params: %s', forward_params)
                    batch_logits = model(**forward_params)[0]
                    # Pull the logits out of torch's graph
                    batch_logits = batch_logits.detach().cpu().numpy()
                    batch_outputs = np.argmax(batch_logits, axis=-1)
                    outputs.append(batch_outputs)
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
        return np.concatenate(outputs, axis=0)
