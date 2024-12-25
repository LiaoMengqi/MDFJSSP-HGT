import subprocess
import re
import numpy as np
import os
import random
import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import warnings


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int = 500,
            T_0: int = 500,
            T_mult: int = 2,
            min_lr: float = 1e-8,
            last_epoch: int = -1,
            eta_min: float = 0,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_lr = min_lr
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            lr_factor = float(epoch) / float(max(1, self.warmup_epochs))
            return [max(self.min_lr, base_lr * lr_factor) for base_lr in self.base_lrs]

        if epoch >= self.warmup_epochs:
            if self.T_mult == 1:
                self.T_cur = epoch - self.warmup_epochs
                cycle = self.T_cur // self.T_0
                self.T_cur = self.T_cur % self.T_0
            else:

                n = int(math.log((epoch - self.warmup_epochs) * (self.T_mult - 1) / self.T_0 + 1, self.T_mult))
                self.T_cur = epoch - self.warmup_epochs - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                cycle = n

            cos_progress = math.cos(math.pi * self.T_cur / self.T_0)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + cos_progress)
                    for base_lr in self.base_lrs]


def warmup_schedule(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def select_gpu():
    try:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode()
    except UnicodeDecodeError:
        nvidia_info = subprocess.run(
            'nvidia-smi', stdout=subprocess.PIPE).stdout.decode("gbk")
    used_list = re.compile(r"(\d+)MiB\s+/\s+\d+MiB").findall(nvidia_info)
    used = [(idx, int(num)) for idx, num in enumerate(used_list)]
    sorted_used = sorted(used, key=lambda x: x[1])
    return sorted_used[0][0]


def set_device(gpu):
    if gpu != -1:
        # use gpu
        if not torch.cuda.is_available():
            # gpu not available
            print('No GPU available. Using CPU.')
            device = 'cpu'
        else:
            # gpu available
            if gpu < -1:
                # auto select gpu
                gpu_id = select_gpu()
                print('Auto select gpu:%d' % gpu_id)
                device = 'cuda:%d' % gpu_id
            else:
                # specify gpu id
                if gpu >= torch.cuda.device_count():
                    gpu_id = select_gpu()
                    print('GPU id is invalid. Auto select gpu:%d' % gpu_id)
                    device = 'cuda:%d' % gpu_id
                else:
                    print('Using gpu:%d' % gpu)
                    device = 'cuda:%d' % gpu
    else:
        print('Using CPU.')
        device = 'cpu'
    return device


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
