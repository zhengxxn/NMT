from torch.optim.lr_scheduler import _LRScheduler


class InverseSqrtLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, init_lr, max_lr, min_lr, last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.lr_linear_step = (max_lr - init_lr) / warmup_steps
        self.decay_factor = max_lr * warmup_steps**0.5
        super(InverseSqrtLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        if last_epoch < self.warmup_steps:
            rate = self.init_lr + last_epoch * self.lr_linear_step
        else:
            rate = self.decay_factor * last_epoch**-0.5
        if rate < self.min_lr:
            rate = self.min_lr

        return [rate
                for _ in self.optimizer.param_groups]
