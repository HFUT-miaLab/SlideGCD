import sys
import datetime
import random

import torch
import numpy as np


class Logger(object):
    def __init__(self, filename='./logs/' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BestModelSaver:
    def __init__(self, max_epoch, ratio=0.3, early_stop=10):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_acc_epoch = 0

        # Only consider selecting the best model after training beyond this round (begin_epoch)
        self.begin_epoch = int(max_epoch * ratio)
        # Parameter for EarlyStop
        self.early_stop_interval = early_stop
        self.early_stop_count = 0

    def update(self, valid_acc, valid_auc, current_epoch):
        if current_epoch < self.begin_epoch:
            return

        if valid_acc > self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_auc = valid_auc
            self.best_valid_acc_epoch = current_epoch
            self.early_stop_count = 0
        elif valid_acc == self.best_valid_acc and valid_auc > self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_acc_epoch = current_epoch
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

    def is_early_stop(self):
        return self.early_stop_count >= self.early_stop_interval


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Fix Random Seeds:", seed)
