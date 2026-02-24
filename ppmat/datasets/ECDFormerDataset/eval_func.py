import math
import json

import paddle
import paddle.nn as nn
import numpy as np

from sklearn.metrics import mean_squared_error

def Accuracy(pred, gt):
    # the implimentation of good sample Accuracy. We calculate multi-range accuracy
    # pred: paddle.Tensor(batch), the tensor of prediction
    # gt:   paddle.Tensor(batch), the tensor of groundtruth
    pred, gt = pred.tolist(), gt.tolist()
    assert len(pred) == len(gt)

    acc, acc_1, acc_2, acc_3 = 0, 0, 0, 0
    for i in range(len(pred)):
        if pred[i] == gt[i]: 
            acc += 1
        elif abs(pred[i]-gt[i]) == 1:
            acc_1 += 1
        elif abs(pred[i]-gt[i]) == 2:
            acc_2 += 1
        elif abs(pred[i]-gt[i]) == 3:
            acc_3 += 1
        else: continue
    
    return acc, acc_1, acc_2, acc_3

def MAE(pred, gt):
    # the implimentation of Mean Absolute Error
    # MAE == L1 loss
    # pred: paddle.Tensor(batch, seq_len), the tensor of prediction
    # gt:   paddle.Tensor(batch, seq_len), the tensor of groundtruth

    pred = paddle.to_tensor(pred, dtype=paddle.float32)
    gt = paddle.to_tensor(gt, dtype=paddle.float32)
    assert pred.shape == gt.shape
    L1Loss = nn.L1Loss(reduction='mean')  # paddle中size_average已废弃，统一用reduction
    mae = L1Loss(pred, gt)

    return mae


def MAPE(pred, gt):
    # the implimentation of Mean Absolute Percentage Error
    # pred: paddle.Tensor(batch, seq_len), the tensor of prediction
    # gt:   paddle.Tensor(batch, seq_len), the tensor of groundtruth

    pred = paddle.to_tensor(pred, dtype=paddle.float32)
    gt = paddle.to_tensor(gt, dtype=paddle.float32)
    pred, gt = pred.numpy(), gt.numpy()  # paddle中无cpu()方法，直接用numpy()
    mape_loss = np.mean(
        np.abs(np.divide(pred-gt, gt, out=np.zeros_like(gt), where=gt!=0))
    ) * 100
    return mape_loss

def get_sequence_peak(sequence):
    # input- seq: List
    # output- peak_list contains peak position
    peak_list = []
    for i in range(1, len(sequence)-1):
        if sequence[i-1]<sequence[i] and sequence[i]>sequence[i+1]:
            peak_list.append(i)
        if sequence[i-1]>sequence[i] and sequence[i]<sequence[i+1]:
            peak_list.append(i)
    return peak_list

def RMSE_Peak(pred, gt):
    # the implementation of RMSE for Peak Number, Position, also the Peak Symbol acc
    # pred: paddle.Tensor(batch, seq_len), the tensor of prediction
    # gt:   paddle.Tensor(batch, seq_len), the tensor of groundtruth
    
    pred = paddle.to_tensor(pred, dtype=paddle.float32)
    gt = paddle.to_tensor(gt, dtype=paddle.float32)
    pred, gt = pred.numpy().tolist(), gt.numpy().tolist()  # paddle中无cpu()方法
    batch_size = len(pred)
    
    # calculate RMSE for Peak number and position
    number_gt, number_pred, position_gt, position_pred = [],[],[],[]
    total_peaks, correct_peak_symbols = 0, 0
    for i in range(batch_size):
        peaks_gt = get_sequence_peak(gt[i])
        peaks_pred = get_sequence_peak(pred[i])
        min_peaks_len,max_peaks_len = min(len(peaks_gt), len(peaks_pred)), max(len(peaks_gt), len(peaks_pred))
        ## peak_height: 
        total_peaks += len(peaks_gt)
        for j in range(min_peaks_len):
            if pred[i][peaks_pred[j]]*gt[i][peaks_gt[j]]>0: correct_peak_symbols += 1
        ## peak number
        number_gt.append(len(peaks_gt))
        number_pred.append(len(peaks_pred))
        ## peak position
        if len(peaks_gt) == min_peaks_len:
            peaks_gt.extend([len(gt[i])] * (max_peaks_len-min_peaks_len) )
        else: peaks_pred.extend([len(pred[i])] * (max_peaks_len-min_peaks_len) )
        position_gt.extend(peaks_gt)
        position_pred.extend(peaks_pred)
        
    rmse_position = np.sqrt(mean_squared_error(position_gt, position_pred))
    rmse_number = np.sqrt(mean_squared_error(number_gt, number_pred))
    symbol_acc = correct_peak_symbols / total_peaks if total_peaks != 0 else 0.0
    return symbol_acc, rmse_position, rmse_number

def Peak_for_draw(pred, gt):
    # the implementation of RMSE for Peak Range, Number, Position
    # return the Peak information
    # pred: paddle.Tensor(batch, seq_len), the tensor of prediction
    # gt:   paddle.Tensor(batch, seq_len), the tensor of groundtruth

    pred = paddle.to_tensor(pred, dtype=paddle.float32)
    gt = paddle.to_tensor(gt, dtype=paddle.float32)
    pred, gt = pred.numpy().tolist(), gt.numpy().tolist()  # paddle中无cpu()方法
    batch_size = len(pred)
    
    # calculate RMSE Range
    range_gt, range_pred = [], []
    for i in range(batch_size):
        range_gt.append(max(gt[i]) - min(gt[i]))
        range_pred.append(max(pred[i]) - min(pred[i]))
    
    # calculate RMSE for Peak number
    number_gt, number_pred = [], []
    position_gt, position_pred = [], []
    for i in range(batch_size):
        peaks_gt = get_sequence_peak(gt[i])
        peaks_pred = get_sequence_peak(pred[i])
        
        number_gt.append(len(peaks_gt))
        number_pred.append(len(peaks_pred))
        
        min_peaks_len = min(len(peaks_gt), len(peaks_pred))
        max_peaks_len = max(len(peaks_gt), len(peaks_pred))
        
        if len(peaks_gt) == min_peaks_len:
            peaks_gt.extend([len(gt[i])] * (max_peaks_len-min_peaks_len) )
        else:
            peaks_pred.extend([len(pred[i])] * (max_peaks_len-min_peaks_len) )
        position_gt.append(sum(peaks_gt))
        position_pred.append(sum(peaks_pred))
        
    return dict(
        peak_range = (range_gt, range_pred),
        peak_num = (number_gt, number_pred),
        peak_pos = (position_gt, position_pred),
    )