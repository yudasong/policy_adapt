# Code for the ICML 2020 paper Provably Efficient Model-based Policy Adaptation

Link to the paper: [Provably Efficient Model-based Policy Adaptation](https://arxiv.org/abs/2006.08051)

This code is partly based on  the [pytorch implementation of PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

## Prerequisites

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

Creating a virtual environment is recommended:
```bash
pip install virtualenv
virtualenv /path/to/venv --python=python3

#To activate a virtualenv: 

. /path/to/venv/bin/activate
```

To install the dependencies (the results from the paper are obtain from gym==0.14.0):
``` bash
pip install -r requirements.txt
```

To install the dependencies from OpenAI baselines:
```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

To install pytorch, please follow [PyTorch](http://pytorch.org/).


## Training


### Source policy
We provided the source policies that are trained using the [pytorch implementation of PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), which are stored in the `policy_adaptation/source_policy/` directory. Since our algorithm is independent from the source policy, you are welcome to try out with other source policies (e.g. TRPO, SAC).

### Source dynamics
We do not provide a trianed dynamics model, but one can always train such a model with the samples collected while training the source policy. We by default use `--original-env` flag to leverage the simulator dynamics.

### Training
We provide example scripts in `example_scripts`. For example,
```
# Performing traning, plotting and testing in HalfCheetah with 120% original mass 
policy_adaptation/halfcheetah_target_mass_12.sh
```

An example of running the training script could be:
```
# Adapting the source policy on a target HalfCheetah environment with 120% of the orginial gravity
python policy_adaptation/dm_main.py --env-name HalfCheetahDM-v2 --original-env --pert-ratio 1.2 --task gravity 
```
One can add the `--train-target` flag to train the target policy. 
The deviation models will be store at `policy_adaptation/trained_models/target_dynamics` and the target policies will be stored at `policy_adaptation/trained_models/target_policies`.

Here is a complete descrition of the training script:

```
python dm_main.py 
--env-name [ENVIRONMENT_ID] #e.g. HalfCheetahDM-v2, AntDM-v2]
--original-env  # Use this flag to use the simulator dynamics
--pert-ratio [RATIO] # A ratio of the perturbation, we use 0.5 to 2.0 in our paper
--train-target # Use this flag to train the target policy
--seed [SEED]
--task [TASK_NAME] # The perturbation name, e.g. mass, gravity, friction
--soft-update-rate [N] # soft update the target policy every N iteration
--num-updates [N] # number of iterations of training
--update-size [N] # size of batch per iteration
--load-dir [DIR] # the direction of the source policies
--pretrain # use this flag to pretrain the DM model, it will result in a faster convergence
--action-noise-std [STD] # the standard deviation of the action noise for the noisy motor task
```

### Plotting and testing
All training data will be saved in `policy_adaptation/result_data`.  
All trained models will be saved in `policy_adaptation/trained_models`.  
Please refer to the example scripts to see the usage of  `plot.py` and  `enjoy.py`. Note that we only provide testing with target policies because testing with deviation model will take a significant amount of time to run.

### Reproducing our results
Please use seed `1,2,3,4,5` if you want to reproduce our results in the paper.


