import torch
from misc.utils import get_render_func, get_vec_normalize
from utils import *
import copy
from cem import run_cem_algorithm



def collect_trajectory(source_dynamics,target_dynamics,policy, target_target_policy, env,original_env,args,
                      random=False, render_func=None, deterministic=False, decay=0):

    recurrent_hidden_states = torch.zeros(1,
                                          policy.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    sasd_list = []
    sp_list = []
    s_list = []
    a_list = []

    if render_func is not None:
        render_func('human')

    obs = env.reset()
    delta_s = obs * 0

    timestep = 0
    episode_rewards = 0

    eps_source_reward = 0

    while True:
        target_dynamics.ob_rms = get_vec_normalize(env).ob_rms

        s_d, a_hat, source_reward = getDesiredState(obs, policy, source_dynamics, target_dynamics, args.env_name, original_env=original_env)
        eps_source_reward += source_reward

        if random:
            action = torch.FloatTensor(np.random.uniform(-1,1,a_hat.shape))
        else:
            recurrent_hidden_states = torch.zeros(obs.shape[0],target_target_policy.recurrent_hidden_state_size)
            masks = torch.zeros(obs.shape[0], 1)
            _, a_hat, _, _ = target_target_policy.act(obs[:,agnostic_size[args.env_name]:], recurrent_hidden_states, masks, deterministic=deterministic)
            action = run_cem_algorithm(args,target_dynamics,obs,s_d,a_hat)

            if (np.random.rand() < args.noise_prob):

                noise = np.random.uniform(-args.noise_range + decay * args.noise_range
                                        ,args.noise_range - decay * args.noise_range,size = action.data.numpy().shape)
                action += torch.FloatTensor(noise)
                '''
                random_act = np.random.uniform(-1, 1, size = action.data.numpy().shape)
                action = torch.FloatTensor(random_act)
                '''
        if args.action_noise_std is not None:
            noise = np.random.normal(0,args.action_noise_std, size = action.data.numpy().shape)
            action += torch.FloatTensor(noise)

        action_hat = copy.deepcopy(action.data)

        last_obs = obs

        obs, reward, done, _ = env.step(action_hat[0])

        timestep += 1
        episode_rewards += reward

        sasd = torch.cat((last_obs,action,s_d),1)

        sasd_list.append(sasd.data.numpy())
        sp_list.append((obs-s_d).data.numpy())
        s_list.append(last_obs.data.numpy())
        a_list.append(action.numpy())

        if render_func is not None:
            render_func('human')

        if done:
            break

    sasd_list = np.asarray(sasd_list)
    sp_list = np.asarray(sp_list)
    s_list = np.asarray(s_list)
    a_list = np.asarray(a_list)

    return sasd_list, sp_list, s_list, a_list, timestep, episode_rewards, eps_source_reward
