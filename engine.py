# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import signal
import time
import subprocess
import matplotlib.pyplot as plt

#def get_gpu_memory_used():
#    try:
#        output = subprocess.check_output(
#            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
#            encoding = "utf-8"
#        )
#        gpu_mem_used = output.strip().split('\n')
#        return int(gpu_mem_used[0])
#    except Exception as e:
#        print(f"Error fetching GPU memory: {e}")
#        return 0

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#    metric_logger.add_meter('max_gpu_mem', utils.SmoothedValue(window_size=1, fmt='{value:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
#    max_gpu_mem_value = 0
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # debug
    # count = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # count += 1
        # if count > 20:
        #     break

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            # outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
       # 在每個 batch 處理完後，取得當前 GPU 使用量
    #    current_gpu_mem = get_gpu_memory_used()
        # 更新最大值
    #    max_gpu_mem_value = max(max_gpu_mem_value, current_gpu_mem)

    metric_logger.update(loss=loss_value)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # metric_logger.update(max_gpu_mem=max_gpu_mem_value)
    


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    #cumulated prob
    pdf = float(np.zeros(5))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            # output : (batch_size * num_class), prob of each class in a batch
            output = model(images)
            loss = criterion(output, target)

        ############
        class_prob = torch_softmax(output, dim=1)
        top_5, _ = class_prob.topk(5, dim=1)
        top_5.sort()
        pdf += top_5
        ############

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    ##########
    pdf = pdf / sum(pdf)

    histogram_path = "/output/acc_histogram.txt"
    x_label = [1, 2, 3, 4, 5]
    plt.bar(x_label, pdf)
    plt.xlabel('Top')
    plt.ylabel('Probability')
    plt.title('Top 5 distribution')
    plt.savefig(histogram_path)
    ##########


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}