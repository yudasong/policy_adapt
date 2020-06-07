import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.distributions import Bernoulli, Categorical, DiagGaussian
from misc.utils import init
import copy

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Dynamics(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Dynamics, self).__init__()

        self.hidden_size = 128

        self.obs_size = obs_shape[0]

        self.ob_rms = None

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = MLPBase
        if action_space.__class__.__name__ == "Discrete":
            num_inputs = action_space.n + obs_shape[0]
            self.action_size = action_space.n
        elif action_space.__class__.__name__ == "Box":
            num_inputs = action_space.shape[0] + obs_shape[0]
            self.action_size = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            num_inputs = action_space.shape[0] + obs_shape[0]
            self.action_size = action_space.shape[0]
        else:
            raise NotImplementedError

        num_outputs = obs_shape[0]
        self.dist = DiagGaussian(self.hidden_size, num_outputs)
        self.base = base(num_inputs,obs_shape[0])

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def predict(self, inputs, deterministic=False):
        hidden = self.base(inputs)
        dist = self.dist(hidden)
        if deterministic:
            sp = dist.mode()
        else:
            sp = dist.sample()
        return  sp

    def evaluate(self, inputs, y):
        hidden = self.base(inputs)
        dist = self.dist(hidden)
        log_prob = dist.log_probs(y)
        return log_prob

    def linear_dynamics(self, cur_traj, cur_actions):
        assert(len(cur_traj) == len(cur_actions))
        T = len(cur_traj)
        sa_len = self.obs_size + self.action_size
        dynamics_m = np.zeros((T, self.obs_size, sa_len))
        dynamics_v = np.zeros((T, self.obs_size))
        i = 0
        for (s,a) in zip(cur_traj, cur_actions):
            sa = torch.cat((s,a),1)
            sa_placeholder = torch.from_numpy(sa.data.numpy())
            sa_placeholder.requires_grad=True
            sp = self.predict(sa_placeholder, deterministic=True)
            for j in range(sp.shape[1]):
                grad, = torch.autograd.grad(sp[0][j],sa_placeholder, create_graph=True)
                dynamics_m[i, j, :] =  grad.data.numpy()
            i += 1
        return dynamics_m, dynamics_v


class MLPBase(nn.Module):
    def __init__(self, num_inputs,num_outputs, hidden_size=128):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.train()

    def forward(self, inputs):
        x = inputs
        hidden_actor = self.actor(x)

        return hidden_actor
