# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb_logger=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.wandb_logger = wandb_logger
        self.iteration = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
        
        # Log to wandb if available
        if self.wandb_logger is not None:
            metrics = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                metrics[k] = v
            self.wandb_logger.log_metrics(metrics, step=self.iteration)

    def log_validation_metrics(self, recall_metrics, iteration):
        """Log validation metrics to wandb"""
        if self.wandb_logger is not None:
            metrics = {
                'val/recall@1': recall_metrics[0],
                'val/recall@2': recall_metrics[1], 
                'val/recall@4': recall_metrics[2],
                'val/recall@8': recall_metrics[3]
            }
            self.wandb_logger.log_metrics(metrics, step=iteration)

    def set_iteration(self, iteration):
        """Set current iteration for wandb logging"""
        self.iteration = iteration

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)
