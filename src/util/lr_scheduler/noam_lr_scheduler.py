from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps, factor, model_size, last_epoch=-1):

        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        rate = self.factor * \
                (self.model_size ** (-0.5) *
                 min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)))
        return [rate
                for _ in self.optimizer.param_groups]
