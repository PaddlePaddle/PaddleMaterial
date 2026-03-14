# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, Optional, Any

import numpy as np
import paddle
from paddle import nn
from paddle import optimizer as optim
from paddle.distributed import fleet

from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.utils import logger
from ppmat.utils import AverageMeter
from ppmat.utils import save_load
from ppmat.metrics.ecd_metric import ECDMetrics
from ppmat.metrics.ir_metric import IRMetrics
from ppmat.losses.ecd_loss import ECDLoss
from ppmat.losses.ir_loss import IRLoss


class ECDFormerTrainer(BaseTrainer):
    """
    ECDFormer trainer supporting both ECD and IR tasks with dedicated metrics.
    
    Features:
    - Automatic task detection from model class name
    - Task-specific loss functions (ECDLoss for classification, IRLoss for regression)
    - Task-specific streaming metrics (ECDMetrics, IRMetrics)
    - Attention visualization during inference
    - Compatible with BaseTrainer training loop
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Layer,
        train_dataloader: Optional[paddle.io.DataLoader] = None,
        val_dataloader: Optional[paddle.io.DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[optim.lr.LRScheduler] = None,
        compute_metric_func_dict: Optional[Dict] = None,
    ):
        # Initialize parent class
        super().__init__(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            compute_metric_func_dict=compute_metric_func_dict,
        )
        
        # Task detection from model class name
        model_class_name = model.__class__.__name__
        self.is_ir_task = "IR" in model_class_name
        self.is_ecd_task = "ECD" in model_class_name
        
        logger.info(f"Task type detected: {'IR' if self.is_ir_task else 'ECD' if self.is_ecd_task else 'Unknown'}")
        
        # Get task-specific parameters from config
        self.max_peaks = config.get("max_peaks", 15 if self.is_ir_task else 9)
        self.num_position_classes = config.get("num_position_classes", 36 if self.is_ir_task else 20)
        
        # Initialize task-specific loss function
        if self.is_ecd_task:
            self.loss_fn = ECDLoss(
                loss_weight_height=config.get("loss_weight_height", 2.0),
                num_position_classes=self.num_position_classes,
                height_classes=config.get("height_classes", 2)
            )
            logger.info("Using ECDLoss for ECD task")
        elif self.is_ir_task:
            self.loss_fn = IRLoss(
                num_position_classes=self.num_position_classes,
                use_height_prediction=config.get("use_height_prediction", True)
            )
            logger.info("Using IRLoss for IR task")
        else:
            # Fallback to simple cross-entropy
            self.ce_loss = nn.CrossEntropyLoss()
            logger.warning("Unknown task type, using fallback CrossEntropyLoss")
        
        # Initialize task-specific metrics (will be attached via attach_metrics)
        self.train_metrics = None
        self.eval_metrics = None
    
    def attach_metrics(self, metric_cfg=None, **runtime_objs):
        """
        Attach task-specific metrics to the trainer.
        
        Args:
            metric_cfg: Metric configuration from config file
            **runtime_objs: Additional runtime objects
        """
        super().attach_metrics(metric_cfg, **runtime_objs)
        
        # Create task-specific metric instances if not already in metric_modules
        if self.is_ecd_task and 'ECDMetrics' not in str(self.metric_modules):
            self.metric_modules['ecd_metrics'] = ECDMetrics(
                num_position_classes=self.num_position_classes,
                max_peaks=self.max_peaks
            )
            logger.info("ECDMetrics attached")
        elif self.is_ir_task and 'IRMetrics' not in str(self.metric_modules):
            self.metric_modules['ir_metrics'] = IRMetrics(
                use_height_prediction=config.get("use_height_prediction", True)
            )
            logger.info("IRMetrics attached")
    
    def train_epoch(self, dataloader: paddle.io.DataLoader):
        """
        Train for one epoch using task-specific loss functions.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            tuple: time_info, loss_info, metric_info
        """
        self.model.train()
        
        # Initialize statistics
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }
        
        # Update training state
        self.state.max_steps_in_train_epoch = len(dataloader)
        self.state.step_in_train_epoch = 0
        
        # Timers
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        
        for iter_id, batch in enumerate(dataloader):
            # Parse batch data (adapting to ECDCollator/IRCollator format)
            model_inputs, targets = batch
            
            reader_cost = time.perf_counter() - reader_tic
            time_info["reader_cost"].update(reader_cost)
            
            # Calculate batch size
            batch_size = model_inputs['x'].shape[0] if hasattr(model_inputs['x'], 'shape') else 1
            
            # Forward pass
            with self.autocast_context_manager(self.use_amp, self.amp_level):
                predictions = self.model(
                    x=model_inputs['x'],
                    edge_index=model_inputs['edge_index'],
                    edge_attr=model_inputs['edge_attr'],
                    batch_data=model_inputs['batch_data'],
                    ba_edge_index=model_inputs.get('ba_edge_index', None),
                    ba_edge_attr=model_inputs.get('ba_edge_attr', None),
                    query_mask=model_inputs.get('query_mask', None)
                )
                
                # Compute loss using task-specific loss function
                loss_dict = self.loss_fn(predictions, targets)
                loss = loss_dict["loss"]
            
            # Backward pass
            if self.use_amp:
                loss_scaled = self.scaler.scale(loss)
                loss_scaled.backward()
            else:
                loss.backward()
            
            # Update parameters
            if self.use_amp:
                self.scaler.minimize(self.optimizer, loss_scaled)
            else:
                self.optimizer.step()
            self.optimizer.clear_grad()
            
            # Update loss statistics
            for key, value in loss_dict.items():
                if key not in loss_info:
                    loss_info[key] = AverageMeter(key)
                loss_info[key].update(float(value), batch_size)
            
            # Update streaming metrics
            self._update_streaming_metrics(result={'predictions': predictions, 'loss_dict': loss_dict}, 
                                           batch=targets, stage='train')
            
            batch_cost = time.perf_counter() - batch_tic
            time_info["batch_cost"].update(batch_cost)
            
            # Update state
            self.state.step_in_train_epoch += 1
            self.state.global_step += 1
            
            # Update learning rate (step-based)
            if self.lr_scheduler is not None and not self.lr_scheduler.by_epoch:
                self.lr_scheduler.step()
            
            # Logging
            if (self.state.step_in_train_epoch % self.log_freq == 0 or
                self.state.step_in_train_epoch == self.state.max_steps_in_train_epoch):
                
                logs = OrderedDict()
                logs["lr"] = self.optimizer.get_lr()
                for name, meter in time_info.items():
                    logs[name] = meter.val
                for name, meter in loss_info.items():
                    logs[name] = meter.val
                
                # Add streaming metrics if available
                stream_metrics = self._compute_streaming_metrics(stage='train')
                for name, value in stream_metrics.items():
                    if isinstance(value, (int, float)):
                        logs[f"{name}"] = value
                        if name not in metric_info:
                            metric_info[name] = AverageMeter(name)
                        metric_info[name].update(float(value), 1)
                
                display_logs = self._filter_out_dict(logs, stage="train")
                
                msg = f"Train: Epoch [{self.state.epoch}/{self.max_epochs}]"
                msg += f" | Step: [{self.state.step_in_train_epoch}/{self.state.max_steps_in_train_epoch}]"
                for key, val in display_logs.items():
                    msg += f" | {key}: {val:.6f}"
                logger.info(msg)
                
                # Write to visualization tools
                logger.scalar(
                    tag="train(step)",
                    metric_dict=logs,
                    step=self.state.global_step,
                    visualdl_writer=self.visualdl_writer,
                    wandb_writer=self.wandb_writer,
                    tensorboard_writer=self.tensorboard_writer,
                )
            
            batch_tic = time.perf_counter()
            reader_tic = time.perf_counter()
        
        # Compute epoch-level streaming metrics
        epoch_stream_metrics = self._compute_streaming_metrics(stage='train')
        for name, value in epoch_stream_metrics.items():
            if isinstance(value, (int, float)):
                if name not in metric_info:
                    metric_info[name] = AverageMeter(name)
                metric_info[name].update(float(value), 1)
        
        return time_info, loss_info, metric_info
    
    def eval_epoch(self, dataloader: paddle.io.DataLoader):
        """
        Evaluate for one epoch using task-specific metrics.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            tuple: time_info, loss_info, metric_info
        """
        self.model.eval()
        
        loss_info = {}
        metric_info = {}
        time_info = {
            "reader_cost": AverageMeter(name="reader_cost", postfix="s"),
            "batch_cost": AverageMeter(name="batch_cost", postfix="s"),
        }
        
        self.state.max_steps_in_eval_epoch = len(dataloader)
        self.state.step_in_eval_epoch = 0
        
        # Reset streaming metrics for evaluation
        for _, m in self.metric_modules.items():
            if hasattr(m, 'reset'):
                m.reset()
        
        reader_tic = time.perf_counter()
        batch_tic = time.perf_counter()
        
        with paddle.no_grad():
            for iter_id, batch in enumerate(dataloader):
                model_inputs, targets = batch
                
                reader_cost = time.perf_counter() - reader_tic
                time_info["reader_cost"].update(reader_cost)
                
                batch_size = model_inputs['x'].shape[0] if hasattr(model_inputs['x'], 'shape') else 1
                
                # Forward pass
                with self.autocast_context_manager(self.use_amp, self.amp_level):
                    predictions = self.model(
                        x=model_inputs['x'],
                        edge_index=model_inputs['edge_index'],
                        edge_attr=model_inputs['edge_attr'],
                        batch_data=model_inputs['batch_data'],
                        ba_edge_index=model_inputs.get('ba_edge_index', None),
                        ba_edge_attr=model_inputs.get('ba_edge_attr', None),
                        query_mask=model_inputs.get('query_mask', None)
                    )
                    
                    # Compute loss
                    loss_dict = self.loss_fn(predictions, targets)
                
                # Update loss statistics
                for key, value in loss_dict.items():
                    if key not in loss_info:
                        loss_info[key] = AverageMeter(key)
                    loss_info[key].update(float(value), batch_size)
                
                # Update streaming metrics
                self._update_streaming_metrics(result={'predictions': predictions, 'loss_dict': loss_dict}, 
                                               batch=targets, stage='eval')
                
                # Step-wise metric computation (if configured)
                if self.metric_strategy_during_eval == "step":
                    step_metrics = self._compute_streaming_metrics(stage='eval')
                    for name, value in step_metrics.items():
                        if isinstance(value, (int, float)):
                            if name not in metric_info:
                                metric_info[name] = AverageMeter(name)
                            metric_info[name].update(float(value), batch_size)
                
                batch_cost = time.perf_counter() - batch_tic
                time_info["batch_cost"].update(batch_cost)
                
                self.state.step_in_eval_epoch += 1
                
                # Logging
                if (self.state.step_in_eval_epoch % self.log_freq == 0 or
                    self.state.step_in_eval_epoch == self.state.max_steps_in_eval_epoch):
                    
                    logs = OrderedDict()
                    for name, meter in time_info.items():
                        logs[name] = meter.val
                    for name, meter in loss_info.items():
                        logs[name] = meter.val
                    
                    display_logs = self._filter_out_dict(logs, stage="eval")
                    
                    msg = f"Eval: Epoch [{self.state.epoch}/{self.max_epochs}]"
                    msg += f" | Step: [{self.state.step_in_eval_epoch}/{self.state.max_steps_in_eval_epoch}]"
                    for key, val in display_logs.items():
                        msg += f" | {key}: {val:.6f}"
                    logger.info(msg)
                
                batch_tic = time.perf_counter()
                reader_tic = time.perf_counter()
        
        # Compute epoch-level metrics from streaming accumulators
        epoch_metrics = self._compute_streaming_metrics(stage='eval')
        for name, value in epoch_metrics.items():
            if isinstance(value, (int, float)):
                if name not in metric_info:
                    metric_info[name] = AverageMeter(name)
                metric_info[name].update(float(value), len(dataloader.dataset))
        
        return time_info, loss_info, metric_info
    
    def predict(self, dataloader: paddle.io.DataLoader) -> Dict[str, Any]:
        """
        Run inference and return predictions with attention visualization.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            dict: Predictions including peak positions, heights, and attention weights
        """
        self.model.eval()
        
        all_pos_pred = []
        all_height_pred = []
        all_attn_weights = []
        all_peak_nums = []
        
        with paddle.no_grad():
            for batch in dataloader:
                model_inputs, _ = batch  # No targets needed for inference
                
                predictions = self.model(
                    x=model_inputs['x'],
                    edge_index=model_inputs['edge_index'],
                    edge_attr=model_inputs['edge_attr'],
                    batch_data=model_inputs['batch_data'],
                    ba_edge_index=model_inputs.get('ba_edge_index', None),
                    ba_edge_attr=model_inputs.get('ba_edge_attr', None),
                    query_mask=model_inputs.get('query_mask', None)
                )
                
                # Get predicted peak numbers
                prob_num = paddle.nn.functional.softmax(predictions['peak_number'], axis=1)
                pred_peak_num = paddle.argmax(prob_num, axis=1)
                all_peak_nums.extend(pred_peak_num.cpu().numpy().tolist())
                
                for i in range(pred_peak_num.shape[0]):
                    n_pred = int(pred_peak_num[i])
                    
                    # Position predictions
                    pos_pred = paddle.argmax(
                        predictions['peak_position'][i, :n_pred, :], axis=1
                    ).cpu().numpy().tolist()
                    
                    # Height predictions (classification or regression)
                    if 'peak_height' in predictions:
                        if len(predictions['peak_height'].shape) == 3:  # Classification (ECD)
                            height_pred = paddle.argmax(
                                predictions['peak_height'][i, :n_pred, :], axis=1
                            ).cpu().numpy().tolist()
                        else:  # Regression (IR)
                            height_pred = predictions['peak_height'][i, :n_pred].reshape([-1]).cpu().numpy().tolist()
                    else:
                        height_pred = []
                    
                    all_pos_pred.append(pos_pred)
                    all_height_pred.append(height_pred)
                    
                    # Attention weights for visualization
                    if predictions.get('attention', {}).get('weights'):
                        all_attn_weights.append({
                            'weights': predictions['attention']['weights'][i],
                            'mask': predictions['attention']['mask'][i] if predictions['attention']['mask'] else None
                        })
        
        return {
            'peak_number': all_peak_nums,
            'peak_position': all_pos_pred,
            'peak_height': all_height_pred,
            'attention': all_attn_weights if all_attn_weights else None
        }
    
    def _update_streaming_metrics(self, *, result, batch, stage: str):
        """
        Update streaming metrics with predictions and targets.
        
        Args:
            result: dict containing 'predictions' from model
            batch: target batch
            stage: 'train' or 'eval'
        """
        predictions = result.get('predictions', {})
        
        for name, metric in self.metric_modules.items():
            if hasattr(metric, 'update'):
                try:
                    metric.update(predictions, batch)
                except Exception as e:
                    logger.debug(f"Error updating metric {name}: {e}")
    
    def _compute_streaming_metrics(self, *, stage: str) -> Dict[str, float]:
        """
        Compute and reset streaming metrics.
        
        Args:
            stage: 'train' or 'eval'
            
        Returns:
            dict: Computed metrics
        """
        all_metrics = {}
        
        for name, metric in self.metric_modules.items():
            if hasattr(metric, 'accumulate'):
                try:
                    metrics = metric.accumulate()
                    if isinstance(metrics, dict):
                        # Add prefix for clarity
                        for k, v in metrics.items():
                            all_metrics[f"{name}/{k}"] = v
                    else:
                        all_metrics[name] = metrics
                except Exception as e:
                    logger.debug(f"Error computing metric {name}: {e}")
            
            if hasattr(metric, 'reset'):
                try:
                    metric.reset()
                except Exception:
                    pass
        
        return all_metrics