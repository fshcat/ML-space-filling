import torch.optim as optim
import numpy as np

class CosinePowerAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, exp_order=10.0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        self.exp_order = exp_order
        self.last_epoch = last_epoch
        self.verbose = verbose

        epochs = np.arange(T_max) + 1
        cos_proportions = (1 + np.cos(np.pi * epochs / T_max)) / 2

        # power curve applied to cosine values
        if exp_order < 1:
            raise ValueError('cosine_power_annealing() requires the "exponent order" parameter '
                             'to be greater than or equal to 1 but got ' + str(exp_order) + '.')
        elif exp_order == 1:
            self.cos_power_proportions = cos_proportions
        else:
            self.cos_power_proportions = np.power(exp_order, cos_proportions + 1)

        self.cos_power_proportions = self.cos_power_proportions - np.min(self.cos_power_proportions)
        self.cos_power_proportions = self.cos_power_proportions / np.max(self.cos_power_proportions)

        super(CosinePowerAnnealingLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]

        return [(base_lr - self.eta_min) * self.cos_power_proportions[self.last_epoch] + self.eta_min
                for base_lr in self.base_lrs]

class CosineAnnealingWithLinearDecay(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, factor=0.5):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.factor = factor
        self.last_restart = 0
        self.cycle = 0
        self.T_cur = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - self.last_restart) == self.T_cur:
            self.last_restart = self.last_epoch
            self.cycle += 1
            self.T_cur = self.T_0 * (self.T_mult ** self.cycle)
            return [self.eta_min + (base_lr * (self.factor ** self.cycle) - self.eta_min) *
                    (1 + np.cos(np.pi / self.T_cur)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr * (self.factor ** self.cycle) - self.eta_min) *
                    (1 + np.cos(np.pi * (self.last_epoch - self.last_restart) / self.T_cur)) / 2
                    for base_lr in self.base_lrs]
def step_scheduler(optimizer, lr_rates, lr_epochs):
    def lr_lambda(epoch):
        idx = 0
        for i in range(len(lr_epochs)):
            if epoch >= lr_epochs[i]:
                idx = i
        return lr_rates[idx] / lr_rates[0]

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

