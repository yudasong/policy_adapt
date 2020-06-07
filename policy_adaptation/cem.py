import parallel_utils
import os
import torch.nn as nn
import torch
import numpy as np

# ================================================================
# Cross-entropy method
# ================================================================

def cem(args,f,th_mean,batch_size,n_iter,elite_frac, initial_std=1.0, extra_std=0.0, std_decay_time=1.0, pool=None):
    r"""
    Noisy cross-entropy method
    http://dx.doi.org/10.1162/neco.2006.18.12.2936
    http://ie.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
    Incorporating schedule described on page 4 (also see equation below.)
    Inputs
    ------
    f : function of one argument--the parameter vector
    th_mean : initial distribution is theta ~ Normal(th_mean, initial_std)
    batch_size : how many samples of theta per iteration
    n_iter : how many iterations
    elite_frac : how many samples to select at the end of the iteration, and use for fitting new distribution
    initial_std : standard deviation of initial distribution
    extra_std : "noise" component added to increase standard deviation.
    std_decay_time : how many timesteps it takes for noise to decay
    \sigma_{t+1}^2 =  \sigma_{t,elite}^2 + extra_std * Z_t^2
    where Zt = max(1 - t / std_decay_time, 10 , 0) * extra_std.
    """
    n_elite = int(np.round(batch_size*elite_frac))

    th_std = np.ones(th_mean.size)*initial_std

    for iteration in range(n_iter):

        extra_var_multiplier = max((1.0-iteration/float(std_decay_time)),0) # Multiply "extra variance" by this factor
        sample_std = np.sqrt(th_std + np.square(extra_std) * extra_var_multiplier)
        ths =  [np.clip(th_mean + dth, -1, 1) for dth in  sample_std[None,:]*np.random.randn(batch_size, th_mean.size)]

        if pool is None:
            ys = np.array([f(th) for th in ths])

        else:
            ys = np.array(pool.map(f, ths))

        assert ys.ndim==1

        elite_inds = ys.argsort()[-n_elite:]
        elite_ths = np.array(ths)[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.var(axis=0).reshape(-1)
        yield {"ys":ys,"th":th_mean,"ymean":ys.mean(), "std" : sample_std}


CEM_OPTIONS = [
    ("batch_size", int, 200, "Number of episodes per batch"),
    ("n_iter", int, 10, "Number of iterations"),
    ("elite_frac", float, 0.2, "fraction of parameter settings used to fit pop"),
    ("initial_std", float, 0.5, "initial standard deviation for parameters"),
    ("extra_std", float, 0.0, "extra stdev added"),
    ("std_decay_time", float, -1.0, "number of timesteps that extra stdev decays over. negative => n_iter/2"),
    ("timestep_limit", int, 0, "maximum length of trajectories"),
    ("parallel", int, 0, "collect trajectories in parallel"),
]

def _cem_objective(th):
    G = parallel_utils.G
    error_criteria = nn.MSELoss()
    sasd = torch.cat((G.state,torch.FloatTensor(th),G.desired_state),1)

    e = G.dynamics.predict(sasd)
    cost = error_criteria(torch.zeros(G.desired_state.shape),e)

    return -cost.data.numpy()

def _seed_with_pid():
    np.random.seed(os.getpid())

def update_default_config(tuples, usercfg):
    """
    inputs
    ------
    tuples: a sequence of 4-tuples (name, type, defaultvalue, description)
    usercfg: dict-like object specifying overrides
    outputs
    -------
    dict2 with updated configuration
    """
    out = dict()
    for (name,_,defval,_) in tuples:
        out[name] = defval
    if usercfg:
        for (k,v) in usercfg.iteritems():
            if k in out:
                out[k] = v #override.
    return out

def run_cem_algorithm(args,dynamics,s_0,s_d,a_hat,usercfg=None, callback=None):
    cfg = update_default_config(CEM_OPTIONS, usercfg)
    if cfg["std_decay_time"] < 0: cfg["std_decay_time"] = cfg["n_iter"] / 2
    #cfg.update(usercfg)
    #print("cem config", cfg)

    G = parallel_utils.G
    G.state = s_0
    G.dynamics = dynamics
    G.desired_state = s_d
    G.exoert_action = a_hat
    if cfg["parallel"]:
        parallel_utils.init_pool()
        pool = G.pool
    else:
        pool = None

    th_mean = a_hat.data.numpy()

    i = 0

    for info in cem(args, _cem_objective, th_mean, cfg["batch_size"], cfg["n_iter"], cfg["elite_frac"],
        cfg["initial_std"], cfg["extra_std"], cfg["std_decay_time"], pool=pool):
        i += 1
    return torch.FloatTensor(info["th"])
