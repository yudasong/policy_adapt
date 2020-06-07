import torch
import numpy as np
import copy
import os
from misc import utils


def getDesiredState(state, policy, source_dynamics,target_dynamics,env_name,original_env=None):

    reward = 0

    if original_env is None:
        state = normalize(source_dynamics.ob_rms, denormalize(target_dynamics.ob_rms,state))

        recurrent_hidden_states = torch.zeros(1,policy.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)
        _, action, _, _ = policy.act(state[:,agnostic_size[env_name]:], recurrent_hidden_states, masks, deterministic=True)

        sa = torch.cat((state,action),1)
        s_prime = source_dynamics.predict(sa,deterministic=True)

        s_prime = normalize(target_dynamics.ob_rms, denormalize(source_dynamics.ob_rms,s_prime))

    else:
        state = denormalize(target_dynamics.ob_rms,state)
        s0 = original_env.reset(init=state.data.numpy())
        recurrent_hidden_states = torch.zeros(1,policy.recurrent_hidden_state_size)
        masks = torch.zeros(1, 1)
        state = normalize(source_dynamics.ob_rms,state)
        _, action, _, _ = policy.act(state[:,agnostic_size[env_name]:], recurrent_hidden_states, masks, deterministic=True)
        obs, reward, done, _ = original_env.step(action.data[0])
        obs = np.asarray([obs])
        obs = torch.from_numpy(obs).float()
        s_prime = normalize(target_dynamics.ob_rms, obs)

    return s_prime, action, reward

def denormalize(ob_rms, state):
    var = ob_rms.var
    #print(var)
    return torch.FloatTensor(state.data.numpy() * np.sqrt(var) + ob_rms.mean)

def normalize(ob_rms, state):
    return torch.FloatTensor((state.data.numpy() - ob_rms.mean) / np.sqrt(ob_rms.var))

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def cap(action, cap_index, cap_value=0.5):
    if torch.sign(action[0][cap_index]) * action[0][cap_index] > cap_value:
        action[0][cap_index] = torch.sign(action[0][cap_index]) * cap_value
    return action

def add_noise(data_inp, noiseToSignal):
    data= copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis = 0)
    std_of_noise = mean_data*noiseToSignal
    for j in range(mean_data.shape[1]):
        if(std_of_noise[0][j]>0):
            data[:,0,j] = np.copy(data[:,0,j]+np.random.normal(0, np.absolute(std_of_noise[0][j]), (data.shape[0],)))
    return data


agnostic_size = {
"HalfCheetahDM-v2": 1,
"AntDM-v2":2
}


def save(target_dynamics, target_target_policy,env, timestep_list, reward_list, max_list, min_list, args):

    if args.train_target:
        save_path = "policy_adaptation/trained_models/target_policy"
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        torch.save(
            [target_target_policy,
            getattr(utils.get_vec_normalize(env), 'ob_rms', None)]
        , os.path.join("policy_adaptation/trained_models/target_policy", "{}_{}_{}_{}.pt".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed))))
    else:
        save_path = "policy_adaptation/trained_models/target_dynamics"
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        torch.save(
            [target_dynamics,
            getattr(utils.get_vec_normalize(env), 'ob_rms', None)]
        , os.path.join("policy_adaptation/trained_models/target_dynamics", "{}_{}_{}_{}.pt".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed))))

    if args.train_target:
        save_path = "policy_adaptation/result_data/target/{}".format(args.env_name+str(args.task))
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        np.save("policy_adaptation/result_data/target/{}/ts_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),timestep_list)
        np.save("policy_adaptation/result_data/target/{}/rw_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),reward_list)
        np.save("policy_adaptation/result_data/target/{}/ma_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),max_list)
        np.save("policy_adaptation/result_data/target/{}/mi_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),min_list)
    else:
        save_path = "policy_adaptation/result_data/cem/{}".format(args.env_name+str(args.task))
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        np.save("policy_adaptation/result_data/cem/{}/ts_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),timestep_list)
        np.save("policy_adaptation/result_data/cem/{}/rw_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),reward_list)
        np.save("policy_adaptation/result_data/cem/{}/ma_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),max_list)
        np.save("policy_adaptation/result_data/cem/{}/mi_{}_{}_{}".format(args.env_name+str(args.task), str(args.pert_ratio), str(args.action_noise_std), str(args.seed)),min_list)
