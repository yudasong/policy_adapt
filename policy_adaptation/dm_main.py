import argparse
import os
import sys
import gym
import numpy as np
import torch
import copy
import torch.optim as optim

from torch.autograd import Variable
import torch.nn as nn

from misc.envs import VecPyTorch, make_vec_envs, make_env
from misc.utils import get_render_func, get_vec_normalize
from collections import deque
from misc.divergence_model import Divergence
from misc.dynamics import Dynamics
from misc.model import Policy

from utils import *

########
from train import update_model
from rollout import collect_trajectory
import env
import os


def run(args):
    args.det = not args.non_det

    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = make_vec_envs(
        args.env_name,
        args.seed,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    env.set_params(args.pert_ratio, args.task)

    # Get a render function
    render_func = get_render_func(env)

    agnostic = agnostic_size[args.env_name]

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name  +".pt"))

    target_target_policy = Policy(
        (env.observation_space.shape[0]-agnostic,),
        env.action_space,
        base_kwargs={'recurrent': False})
    copy_model_params_from_to(actor_critic, target_target_policy)

    if args.train_target:
        target_policy = Policy(
            (env.observation_space.shape[0]-agnostic,),
            env.action_space,
            base_kwargs={'recurrent': False})

    if args.original_env:
        original_env = gym.make(args.env_name)
        original_env.seed(args.seed)
        source_dynamics = Dynamics(env.observation_space.shape,
                                env.action_space)
        source_dynamics.ob_rms = copy.deepcopy(ob_rms)

    else:
        original_env = None
        source_dynamics = torch.load(os.path.join("./trained_models/mb/", args.env_name + ".pt"))
        source_dynamics.ob_rms = copy.deepcopy(ob_rms)

    target_dynamics = Divergence(
        env.observation_space.shape,
        env.action_space)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    SA_buffer = deque(maxlen=args.update_size * 10)
    SP_buffer = deque(maxlen=args.update_size * 10)
    S_buffer = deque(maxlen=args.update_size * 10)
    A_buffer = deque(maxlen=args.update_size * 10)

    rewards_buffer = deque(maxlen=3)

    timestep = 0

    timestep_list = []
    reward_list = []
    max_list = []
    min_list = []

    if args.pretrain:
	    for i in range(5):

	        SAs, SPs, Ss, As, ts, rs, srs = collect_trajectory(source_dynamics,target_dynamics, actor_critic, target_target_policy,env, original_env, args, random=True)
	        SA_buffer.extend(SAs)
	        SP_buffer.extend(SPs)
	        SAs = np.asarray(list(SA_buffer))
	        SPs = np.asarray(list(SP_buffer))
	        loss = update_model(SAs, SPs, target_dynamics, agnostic)



    SA_buffer = deque(maxlen=args.update_size * 10)
    SP_buffer = deque(maxlen=args.update_size * 10)

    for i in range(args.num_updates):
        cur_rewards = []
        cur_len = 0

        while cur_len < args.update_size:
            if (i+1) % 5 is not 0:
                SAs, SPs, Ss, As, ts, rs, srs = collect_trajectory(source_dynamics,
                        target_dynamics, actor_critic,target_target_policy,env, original_env, args, decay=0)
            else:
                SAs, SPs, Ss, As, ts, rs, srs = collect_trajectory(source_dynamics,target_dynamics,actor_critic,
                        target_target_policy,env,original_env,args, deterministic=True, decay= 0)


            SA_buffer.extend(SAs)
            SP_buffer.extend(SPs)
            S_buffer.extend(Ss)
            A_buffer.extend(As)

            cur_len += len(As)
            timestep += ts
            rewards_buffer.extend(rs)
        timestep_list.append(timestep)

        rew = np.mean(np.asarray(list(rewards_buffer)))
        reward_list.append(rew)
        rew_max = np.max(np.asarray(list(rewards_buffer)))
        rew_min = np.min(np.asarray(list(rewards_buffer)))


        SAs = np.asarray(list(SA_buffer))
        SPs = np.asarray(list(SP_buffer))
        Ss = np.asarray(list(S_buffer))
        As = np.asarray(list(A_buffer))

        max_list.append(rew_max)
        min_list.append(rew_min)

        print("iter {}: average reward: {}, max reward: {}".format(i, rew, rew_max))

        print("updating target dynamics model...")
        loss = update_model(SAs, SPs, target_dynamics, agnostic, decay= i / float(args.num_updates), num_epoch=5)
        print("loss: {}".format(loss))

        if args.train_target:
            print("updating policy...")
            loss = update_model(Ss, As, target_policy, agnostic, decay= i / float(args.num_updates), is_policy=True, num_epoch=5)
            print("loss: {}".format(loss))

        if (i % args.soft_update_rate == 0) and i != 0:
            if args.train_target:
                soft_update_from_to(target_policy, target_target_policy,
                        args.lower_alpha + (i / float(args.num_updates))* (args.upper_alpha - args.lower_alpha))

        save(target_dynamics, target_target_policy,env, timestep_list, reward_list, max_list, min_list, args)


def main():
    sys.path.append('misc')

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--env-name',
        default='HalfCheetah-v2',
        help='environment to train on (default: HalfCheetah-v2)')
    parser.add_argument(
        '--load-dir',
        default='policy_adaptation/source_policy/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    parser.add_argument('--pert-ratio', type=float, default=1.0)
    parser.add_argument('--num-updates',type=int,default=80)
    parser.add_argument('--update-size',type=int,default=1000)
    parser.add_argument('--show',type=bool,default=False)
    parser.add_argument('--cuda',type=bool,default=False)
    parser.add_argument('--noise-prob', type=float, default=0.01)
    parser.add_argument('--noise-range', type=float, default=0.1)

    parser.add_argument('--action-noise-std', type=float, default=None)
    parser.add_argument("--soft-update-rate", type=int, default=3)
    parser.add_argument("--task", type=str, default=None)

    parser.add_argument(
        '--pretrain',
        action='store_true',
        default=False)

    parser.add_argument(
        '--train-target',
        action='store_true',
        default=False,
        help='train a target policy')

    parser.add_argument(
        '--original-env',
        action='store_true',
        default=False,
        help='train a target policy')

    parser.add_argument("--upper-alpha", type=float, default=0.2)
    parser.add_argument("--lower-alpha", type=float, default=0.05)

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
