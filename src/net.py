import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from numpy.random import default_rng


def to_rgba(x):
    """returns the four first values of last dimension"""
    return x[..., :4]


def get_living_mask(x):
    """returns boolean vector of the same shape as x, except for the last dimension.
    The last dimension is a single value, true/false, that determines if alpha > 0.1"""
    alpha = x[:, :, :, 3:4]
    m = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
    return m


class CAModel(nn.Module):
    """Neural Network used as update rules for ES and Hebbian."""
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

        # Initialize perception kernel
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
        
        self.double()

    def perceive(self, x):
        """Percieve neighboors with two sobel filters and one single-entry filter"""
        y = F.conv2d(x.permute(0, 3, 1, 2), self.kernel, groups=16, padding=1)
        y = y.permute(0, 2, 3, 1)
        return y

    def forward(self, x, fire_rate=None, step_size=1.0):
        """Forward a cell grid through the network and return the cell grid with changes applied."""
        y = self.perceive(x)
        pre_life_mask = get_living_mask(x)
        dx1 = self.fc0(y)
        # dx1 = torch.tanh(dx1)
        dx1 = F.relu(dx1)
        dx2 = self.fc1(dx1)
        dx = dx2 * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate

        update_mask_rand = torch.rand(*x[:, :, :, :1].shape)
        update_mask = update_mask_rand <= fire_rate
        x += dx * update_mask.double()
        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask.bool() & post_life_mask.bool()
        res = x * life_mask.double()

        if self.use_hebb:
            return y, dx1, res
        else:
            return res

    def loss_f(self, x, y):
        """Return mean squared error between x and y"""
        return torch.mean(torch.square(x[..., :4] - y), [-2, -3, -1])
