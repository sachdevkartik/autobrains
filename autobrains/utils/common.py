import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


class AverageMeterLoss(AverageMeter):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.last_best = self.best = float("inf")
        super().reset()

    def update(self, val):
        super().update(val)
        if self.avg < self.best:
            self.last_best = self.best
            self.best = self.avg


class AverageMeterAccuracy(AverageMeter):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.last_best = self.best = float("-inf")
        super().reset()

    def update(self, val):
        if val > self.best:
            self.last_best = self.best
            self.best = val
        super().update(val)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, targets):
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse)
        return rmse
