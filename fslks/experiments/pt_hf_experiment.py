import typing

import numpy as np
import tensorflow as tf
import torch
import torch.nn
import tqdm.auto as tqdm
import transformers
from absl import logging

from fslks.experiments import Experiment

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
                  epsilon: float = 1e-8) -> torch.optim.Optimizer:
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
    return optimizer


class PTExperiment(Experiment[transformers.PreTrainedModel]):

    def __init__(self,
                 tokenizer_name: str,
                 data_dir: str,
                 # checksum_dir: str,
                 max_seq_len: int,
                 max_grad_norm: int = 1,
                 gradient_accumulation_steps: int = 1,
                 use_amp: bool = True):
        super().__init__(tokenizer_name=tokenizer_name, data_dir=data_dir, max_seq_len=max_seq_len)
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

    def train(self,
              model: transformers.PreTrainedModel,
              tasks: typing.List[str],
              num_epochs: int,
              batch_size: int,
              steps_per_epoch: int,
              prefetch_size: int,
              eval_batch_size: typing.Optional[int] = None,
              eval_batches: typing.Optional[int] = None,
              checkpoint_file: typing.Optional[str] = None) -> None:
        logging.info('Preparing kitchen sink with %d tasks: %s', len(tasks), tasks)

        # Train the model & return its training history
        logging.info('Beginning training...')
        training_data = self.load_train_data(tasks,
                                             batch_size=batch_size,
                                             prefetch_size=prefetch_size).as_numpy_iterator()

        validation_data = self.load_valid_data(tasks,
                                               batch_size=eval_batch_size or batch_size,
                                               prefetch_size=prefetch_size,
                                               num_batches=eval_batches).as_numpy_iterator()

        opt = get_optimizer(model)

        if self.use_amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model: transformers.PreTrainedModel
            opt: torch.optim.Optimizer
            model, opt = amp.initialize(model, opt, opt_level='O1')

        warmup_epochs = 3
        num_epochs += warmup_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=warmup_epochs * steps_per_epoch,
            num_training_steps=num_epochs * steps_per_epoch
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        model.to(self.device)
        model.zero_grad()
        global_step = 0
        epoch_itr = tqdm.trange(0, num_epochs * steps_per_epoch, desc="Training", )
        for epoch in range(1, num_epochs):
            running_loss = 0.

            training_itr = tqdm.tqdm(training_data, desc="Epoch %d" % epoch, initial=1, leave=True, unit=" steps")
            for step, (inputs, labels, _) in enumerate(training_itr, 1):
                epoch_itr.update()
                inputs = {k: torch.from_numpy(v).to(device=self.device, dtype=INPUT_DTYPES[k])
                          for k, v in inputs.items()}
                labels = torch.from_numpy(np.squeeze(labels)).to(device=self.device, dtype=torch.long)
                model.train()

                # Run the forward pass
                outputs = model(input_ids=inputs.get('input_ids'),
                                attention_mask=inputs.get('attention_mask'),
                                token_type_ids=inputs.get('token_type_ids'),
                                position_ids=inputs.get('position_ids'),
                                head_mask=inputs.get('head_mask'),
                                labels=labels)
                loss = outputs[0]

                # Run the backwards pass
                if self.use_amp:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                running_loss += loss.item()
                if step % self.gradient_accumulation_steps == 0:
                    params = amp.master_params(opt) if self.use_amp else model.parameters()
                    torch.nn.utils.clip_grad_norm(params, self.max_grad_norm)
                    opt.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                if step == steps_per_epoch:
                    break

            training_itr.set_postfix_str('Global step: %d, loss: %f' % (global_step, running_loss / global_step))
            training_itr.close()
            logging.get_absl_handler()
        epoch_itr.close()

    def predict_task_split(self, model: transformers.PreTrainedModel, inputs: tf.data.Dataset) -> typing.Optional[
        np.ndarray]:
        try:
            outputs = []
            model.eval()
            for batch in inputs.as_numpy_iterator():
                with torch.no_grad():
                    batch_inputs = torch.from_numpy(batch['input_ids']).to(device=self.device, dtype=torch.long)
                    batch_outputs = model(input_ids=batch_inputs)
                    batch_outputs = batch_outputs[0].detach().cpu().numpy()
                    batch_outputs = np.argmax(batch_outputs, axis=-1)
                    # batch_outputs = np.asarray([outputs.detach().cpu().numpy() for outputs in batch_outputs])
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
