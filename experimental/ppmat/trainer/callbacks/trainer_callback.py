# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is heavily adapted from：
# https://github.com/huggingface/transformers/tree/main/src/transformers/callbacks.py
"""
Callbacks to use with the Trainer class and customize the training loop.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Dict
from typing import Optional

from ppmat.utils import logger


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model
    and optimizer when checkpointing and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using
    gradient accumulation, one update step may require several forward and backward
    passes: if you use `gradient_accumulation_steps=n`, then one update step requires
    going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the
            decimal part being the percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            When tracking the inputs tokens, the number of tokens seen during training
            (number of input tokens, not the number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the
            beginning of training (stored as floats to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so
            far.
        best_global_step (`int`, *optional*):
            When tracking the best model, the step at which the best metric was
            encountered. Used for setting `best_model_checkpoint`.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for
            the best model encountered so far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training
            in a distributed fashion on several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a
            distributed fashion on several machines, this is only going to be `True`
            for one process).
        stateful_callbacks (`List[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved
            or restored. Relevant callbacks should implement a `state` and
            `from_state` function.
    """

    # training state, updated during training
    mode: str = "train"
    epoch: int = 0
    global_step: int = 0
    step_in_train_epoch: int = 0
    step_in_eval_epoch: int = 0
    step_in_test_epoch: int = 0

    max_steps_in_train_epoch: int = 0
    max_steps_in_eval_epoch: int = 0
    max_steps_in_test_epoch: int = 0

    log_history: list[dict[str, float]] = None

    cur_metric: Optional[float] = None
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None

    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    stateful_callbacks: list["TrainerCallback"] = None

    @property
    def step_in_epoch(self):
        mode = self.mode
        if mode == "train":
            step = self.step_in_train_epoch
        elif mode == "eval":
            step = self.step_in_eval_epoch
        elif mode == "test":
            step = self.step_in_test_epoch
        else:
            raise ValueError(
                f'mode must be one of ["train", "eval", "test", but got {mode}'
            )
        return step

    @property
    def max_steps_in_epoch(self):
        mode = self.mode
        if mode == "train":
            step = self.max_steps_in_train_epoch
        elif mode == "eval":
            step = self.max_steps_in_eval_epoch
        elif mode == "test":
            step = self.max_steps_in_test_epoch
        else:
            raise ValueError(
                f'mode must be one of ["train", "eval", "test", but got {mode}'
            )
        return step

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            # We are loading the callbacks in from the state file, no need to process
            # them
            pass
        else:
            # Saveable callbacks get stored as dict of kwargs
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, (ExportableState)):
                    raise TypeError(
                        "All callbacks passed to be saved must inherit "
                        f"`ExportableState`, but received {type(callback)}"
                    )
                name = callback.__class__.__name__
                if name in stateful_callbacks:
                    # We can have multiple versions of the same callback
                    # if so, we store them as a list of states to restore
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(cls, dict_data):
        return cls(**dict_data)

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = (
            json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))

    def compute_steps(self, config, max_steps):
        """
        Calculates and stores the absolute value for logging,
        eval, and save steps based on if it was a proportion
        or not.
        """
        for step_kind in ("logging", "eval", "save"):
            num_steps = getattr(config, f"{step_kind}_steps")
            if num_steps is not None:
                if num_steps < 1:
                    num_steps = math.ceil(max_steps * num_steps)
                setattr(self, f"{step_kind}_steps", num_steps)


class ExportableState:
    """
    A class for objects that include the ability to have its state
    be saved during `Trainer._save_checkpoint` and loaded back in during
    `Trainer._load_from_checkpoint`.

    These must implement a `state` function that gets called during the respective
    Trainer function call. It should only include parameters and attributes needed to
    recreate the state at a particular time, to avoid utilizing pickle/maintain standard
    file IO writing.

    """

    def state(self) -> dict:
        raise NotImplementedError(
            "You must implement a `state` function to utilize this class."
        )

    @classmethod
    def from_state(cls, state):
        instance = cls(**state["args"])
        for k, v in state["attributes"].items():
            setattr(instance, k, v)
        return instance


@dataclass
class TrainerControl(ExportableState):
    """
    A class that handles the [`Trainer`] control flow. This class is used by the
    [`TrainerCallback`] to activate some switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will
            just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of
            the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of
            the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of
            the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of
            the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save_by_freq: bool = False
    should_save_latest: bool = False
    should_save_best: bool = False

    should_evaluate: bool = False
    should_log_step: bool = False
    should_log_epoch: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False
        self.should_save_by_freq = False
        self.should_save_latest = False
        self.should_save_best = False

        self.should_evaluate = False
        self.should_log_epoch = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_log_step = False

    def state(self) -> dict:
        return {
            "args": {
                "should_training_stop": self.should_training_stop,
                "should_epoch_stop": self.should_epoch_stop,
                "should_save_by_freq": self.should_save_by_freq,
                "should_save_latest": self.should_save_latest,
                "should_save_best": self.should_save_best,
                "should_evaluate": self.should_evaluate,
                "should_log_step": self.should_log_step,
                "should_log_epoch": self.should_log_epoch,
            },
            "attributes": {},
        }


class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events
    and take some decisions. At each of those events the following arguments are
    available:

    Args:
        args ([`Dict`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some
            decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data. This is deprecated in favour of
            `processing_class`.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which
    case the event that changes it should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the
    others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an
    example, see the code of the simple [`~transformers.PrinterCallback`].

    """

    def on_init_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.
        """
        pass

    def on_eval_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an evaluation phase.
        """
        pass

    def on_eval_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an evaluation phase.
        """
        pass

    def on_test_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of testing.
        """
        pass

    def on_test_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of testing.
        """
        pass

    def on_train_epoch_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_train_epoch_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_eval_epoch_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an evaluation epoch.
        """
        pass

    def on_eval_epoch_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an evaluation epoch.
        """
        pass

    def on_test_epoch_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a test epoch.
        """
        pass

    def on_test_epoch_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a test epoch.
        """
        pass

    def on_train_step_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient
        accumulation, one training step might take several inputs.
        """
        pass

    def on_train_step_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one
        training step might take several inputs.
        """
        pass

    def on_eval_step_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an evaluation step. If using gradient
        accumulation, one evaluation step might take several inputs.
        """
        pass

    def on_eval_step_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an evaluation step. If using gradient accumulation,
        one evaluation step might take several inputs.
        """
        pass

    def on_test_step_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a test step. If using gradient accumulation,
        one test step might take several inputs.
        """
        pass

    def on_test_step_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a test step. If using gradient accumulation, one
        test step might take several inputs.
        """
        pass

    def on_pre_optimizer_step(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called before the optimizer step but after gradient clipping. Useful for
        monitoring gradients.
        """
        pass

    def on_optimizer_step(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after the optimizer step but before gradients are zeroed out.
        Useful for monitoring gradients.
        """
        pass

    def on_train_substep_begin(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of an substep during gradient accumulation.
        """
        pass

    def on_train_substep_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_save_by_freq(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_save_latest(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_save_best(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log_step(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after logging the last logs.
        """
        pass

    def on_log_epoch(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called after logging the last logs.
        """
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a "
                "`DefaultFlowCallback` in its callbacks. You should add one before "
                "training with `trainer.add_callback(DefaultFlowCallback). The current "
                "list of callbacks is\n:" + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but "
                "there is already one. The current list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, config: Dict, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", config, state, control)

    def on_train_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_training_stop = False
        return self.call_event("on_train_begin", config, state, control)

    def on_train_end(self, config: Dict, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", config, state, control)

    def on_eval_begin(self, config: Dict, state: TrainerState, control: TrainerControl):
        control.should_evaluate = False
        return self.call_event("on_eval_begin", config, state, control)

    def on_eval_end(self, config: Dict, state: TrainerState, control: TrainerControl):
        return self.call_event("on_eval_end", config, state, control)

    def on_test_begin(self, config: Dict, state: TrainerState, control: TrainerControl):
        return self.call_event("on_test_begin", config, state, control)

    def on_test_end(self, config: Dict, state: TrainerState, control: TrainerControl):
        return self.call_event("on_test_end", config, state, control)

    def on_train_epoch_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_epoch_stop = False
        return self.call_event("on_train_epoch_begin", config, state, control)

    def on_train_epoch_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_epoch_end", config, state, control)

    def on_eval_epoch_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_epoch_stop = False
        return self.call_event("on_eval_epoch_begin", config, state, control)

    def on_eval_epoch_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_eval_epoch_end", config, state, control)

    def on_test_epoch_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_epoch_stop = False
        return self.call_event("on_test_epoch_begin", config, state, control)

    def on_test_epoch_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_test_epoch_end", config, state, control)

    def on_train_step_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_step_begin", config, state, control)

    def on_train_step_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_step_end", config, state, control)

    def on_eval_step_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_eval_step_begin", config, state, control)

    def on_eval_step_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_eval_step_end", config, state, control)

    def on_test_step_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_test_step_begin", config, state, control)

    def on_test_step_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_test_step_end", config, state, control)

    def on_train_substep_begin(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_substep_begin", config, state, control)

    def on_train_substep_end(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_train_substep_end", config, state, control)

    def on_pre_optimizer_step(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_pre_optimizer_step", config, state, control)

    def on_optimizer_step(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        return self.call_event("on_optimizer_step", config, state, control)

    def on_save_by_freq(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_save_by_freq = False
        return self.call_event("on_save_by_freq", config, state, control)

    def on_save_latest(
        self, config: Dict, state: TrainerState, control: TrainerControl
    ):
        control.should_save_latest = False
        return self.call_event("on_save_latest", config, state, control)

    def on_save_best(self, config: Dict, state: TrainerState, control: TrainerControl):
        control.should_save_best = False
        return self.call_event("on_save_best", config, state, control)

    def on_log_step(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        logs,
    ):
        control.should_log_step = False
        return self.call_event("on_log_step", config, state, control, logs=logs)

    def on_log_epoch(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        logs,
    ):
        control.should_log_epoch = False
        return self.call_event("on_log_epoch", config, state, control, logs=logs)

    def call_event(self, event, config, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                config,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs,
    evaluation and checkpoints.
    """

    def on_eval_begin(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        state.mode = "eval"
        return control

    def on_test_begin(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        state.mode = "test"
        return control

    def on_train_epoch_end(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if (
            state.epoch % config["save_freq"] == 0
            or state.epoch == config["max_epochs"]
            or state.epoch == 1
        ):
            control.should_save_by_freq = True
        control.should_save_latest = True
        control.should_log_epoch = True
        if (
            state.epoch % config["eval_freq"] == 0
            or state.epoch == config["max_epochs"]
            or state.epoch == 1
        ):
            control.should_evaluate = True
        return control

    def on_eval_epoch_end(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if state.best_epoch == state.epoch:
            control.should_save_best = True

        control.should_log_epoch = True

        return control

    def on_test_epoch_end(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        control.should_log_epoch = True
        return control

    def on_train_step_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):

        if (
            state.step_in_train_epoch % config["log_freq"] == 0
            or state.step_in_train_epoch == state.max_steps_in_train_epoch
            or state.step_in_train_epoch == 1
        ):
            control.should_log_step = True
        return control

    def on_eval_step_end(
        self,
        config: Dict,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (
            state.step_in_eval_epoch % config["log_freq"] == 0
            or state.step_in_eval_epoch == state.max_steps_in_eval_epoch
            or state.step_in_eval_epoch == 1
        ):
            control.should_log_step = True
        return control

    def on_test_step_end(
        self, config: dict, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if (
            state.step_in_test_epoch % config["log_freq"] == 0
            or state.step_in_test_epoch == state.max_steps_in_test_epoch
            or state.step_in_test_epoch == 1
        ):
            control.should_log_step = True
        return control

    def on_log_step(self, config, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            mode = state.mode
            max_epochs = config["max_epochs"]
            epoch = state.epoch
            step_in_epoch = state.step_in_epoch
            max_steps_in_epoch = state.max_steps_in_epoch

            msg = f"{mode}: Epoch [{epoch}/{max_epochs}]"
            msg += f" | Step: [{step_in_epoch}/{max_steps_in_epoch}]"
            if mode == "train" and kwargs.get("optimizer", None) is not None:
                optimizer = kwargs.get("optimizer", None)
                msg += f"| lr: {optimizer.get_lr():.6f}"
            if logs is not None:
                for key, val in logs.items():
                    msg += f" | {key}: {val:.6f}"
            logger.info(msg)

    def on_log_epoch(self, config, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            mode = state.mode
            max_epochs = config["max_epochs"]
            epoch = state.epoch

            msg = f"{mode}: Epoch [{epoch}/{max_epochs}]"
            if logs is not None:
                for key, val in logs.items():
                    msg += f" | {key}: {val:.6f}"
            logger.info(msg)
