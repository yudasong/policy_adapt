import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import seaborn as sns
import env
import os

mpl.style.use('seaborn')
sns.set_color_codes("deep")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--env-name',
    default='HalfCheetahDM-v2')

parser.add_argument(
    '--task',
    type=str,
    default='mass')

parser.add_argument(
    '--pert-ratio',
    type=float,
    default=1.0)

parser.add_argument(
    '--action-noise-std',
    type=float,
    default=None)
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument(
    '--train-target',
    action='store_true',
    default=False,
    help='train a target policy')

args = parser.parse_args()

fig, ax = plt.subplots()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(30)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)

#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
if args.train_target:
    m = "target"
else:
    m = "cem"

if args.action_noise_std is not None:
    args.task = "None"

rw_list = np.load("policy_adaptation/result_data/{}/{}/rw_{}_{}_{}.npy".format(m,args.env_name+args.task, str(args.pert_ratio), str(args.action_noise_std),str(args.seed)))
ts_list = np.load("policy_adaptation/result_data/{}/{}/ts_{}_{}_{}.npy".format(m,args.env_name+args.task, str(args.pert_ratio), str(args.action_noise_std),str(args.seed)))

ax.plot(ts_list, rw_list)

ax.set_xlabel("Number of Timesteps")
ax.set_ylabel("Rewards")

#legend = ax.legend(loc='lower right')
if args.action_noise_std is not None:
    args.task = "action noise"
    args.pert_ratio = args.action_noise_std
plt.title("{}_{}_{}".format(args.env_name, args.task, str(args.pert_ratio)),fontsize=30)

save_path = "policy_adaptation/plots"
try:
    os.makedirs(save_path)
except OSError:
    pass

plt.savefig("policy_adaptation/plots/{}_{}_{}.pdf".format(args.env_name, args.task, str(args.pert_ratio)),bbox_inches='tight')
