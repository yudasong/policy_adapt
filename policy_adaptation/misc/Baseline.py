import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.distributions import Bernoulli, Categorical, DiagGaussian
from misc.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Baseline(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Baseline, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase

        self.hidden_size  = 64

        num_outputs = 1
        if action_space.__class__.__name__ == "Discrete":
            num_inputs = action_space.n + obs_shape[0] * 2
        elif action_space.__class__.__name__ == "Box":
            num_inputs = action_space.shape[0] + obs_shape[0] * 2
        elif action_space.__class__.__name__ == "MultiBinary":
            num_inputs = action_space.shape[0] + obs_shape[0] * 2
        else:
            raise NotImplementedError

        #num_inputs =obs_shape[0]

        self.base = base(num_inputs,num_outputs)

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def evaluate(self, inputs):
        b = self.base(inputs)
        return  b

class MLPBase(nn.Module):
    def __init__(self, num_inputs,num_outputs, hidden_size=64):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, 1)))
        self.train()

    def forward(self, inputs):
        x = inputs
        b = self.actor(x)

        return b
