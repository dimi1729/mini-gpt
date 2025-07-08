import math
import torch

class LearningRateScheduler():
    def __init__(self, optimizer: torch.optim.Optimizer,
                 max_lr: float = 6e-4,
                 min_lr: float = 6e-5,
                 warmup_steps: int = 50,
                 max_steps: int = 1000
                 ):
        """Manual implementation of LR Scheduler with linear warmup + cosine decay"""
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, current_step: int):
        if current_step < self.warmup_steps:
            return self.max_lr * (current_step + 1) / self.warmup_steps

        if current_step >= self.max_steps:
            return self.min_lr  # After max steps keep learning rate const

        decay_ratio = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1, f"Decay ratio {decay_ratio} out of bounds"
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + (self.max_lr - self.min_lr) * coeff

    def set_lr(self, current_step: int):
        lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
