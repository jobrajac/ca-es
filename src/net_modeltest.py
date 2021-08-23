import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from numpy.random import default_rng


"""
This file is very similar to net.py, with minor changes to return more information.
See the comments on net.py for method descriptions.
"""

def to_rgba(x):
    return x[..., :4]

def get_living_mask(x):
    alpha = x[:, :, :, 3:4]
    m = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
    return m


class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, hidden_size, new_size_pad, batch_size, disable_grad=True, use_hebb=False):
        super(CAModel, self).__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.new_size_pad = new_size_pad
        self.batch_size = batch_size
        self.use_hebb = use_hebb

        self.fc0 = nn.Linear(self.channel_n * 3, self.hidden_size, bias=False)
        self.fc1 = nn.Linear(self.hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        identify = np.float64([0, 1, 0])
        identify = np.outer(identify, identify)
        identify = torch.from_numpy(identify)
        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dx = torch.from_numpy(dx)
        dy = dx.T
        self.kernel = torch.cat([identify[None, None, ...], dx[None, None, ...], dy[None, None, ...]], dim=0).repeat(self.channel_n, 1, 1, 1)

        if disable_grad:
            for param in self.parameters():
                param.requires_grad = False
        else: 
            for param in self.parameters():
                param.requires_grad = True
        self.double()

    def perceive(self, x):
        x = F.conv2d(x.permute(0, 3, 1, 2), self.kernel, groups=16, padding=1)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, x, fire_rate=None, step_size=1.0):
        pre_life_mask = get_living_mask(x)
        x_old = x.detach().clone()
        x = self.perceive(x)
        if self.use_hebb:
            y = x.detach().clone()
        x = self.fc0(x)
        x = F.relu(x)
        if self.use_hebb:
            dx1 = x.detach().clone()
        x = self.fc1(x)
        x = x * step_size
        dx = x.detach().clone()
        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask_rand = torch.rand(*x[:, :, :, :1].shape)
        update_mask = update_mask_rand <= fire_rate
        x = x_old + x * update_mask.double()

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask.bool() & post_life_mask.bool()
        x = x * life_mask.double()
        x = x.reshape(self.batch_size, self.new_size_pad, self.new_size_pad, self.channel_n)
        if self.use_hebb:
            return y, dx1, x, dx
        else:
            return x, dx

    def loss_f(self, x, y):
        return torch.mean(torch.square(x[..., :4] - y), [-2, -3, -1])
