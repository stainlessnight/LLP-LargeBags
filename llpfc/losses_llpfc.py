import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.constraints import simplex


def loss_f(x, y, weights, epsilon=1e-8):
    assert torch.all(simplex.check(x))
    x = torch.clamp(x, epsilon, 1 - epsilon)
    unweighted = nn.functional.nll_loss(torch.log(x), y, reduction='none')
    weights /= weights.sum()
    return (unweighted * weights).sum()

def loss_f_val(x, y, weights, epsilon=1e-8):
    assert torch.all(simplex.check(x))
    x = torch.clamp(x, epsilon, 1 - epsilon)
    unweighted = nn.functional.nll_loss(torch.log(x), y, reduction='none')
    return (unweighted * weights).sum()

def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')

class LLPFCLoss_val(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target, weight):
        loss = loss_f_val(input, target, weight, epsilon=1e-8)
        return loss

class LLPFCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input, target, weight):
        loss = loss_f(input, target, weight, epsilon=1e-8)
        return loss
