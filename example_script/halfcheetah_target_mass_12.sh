#python policy_adaptation/dm_main.py --env-name HalfCheetahDM-v2 --original-env --pert-ratio 1.2 --task mass --train-target --seed 1
python policy_adaptation/plot.py --env-name HalfCheetahDM-v2 --pert-ratio 1.2 --task mass --seed 1 --train-target
python policy_adaptation/enjoy.py --env-name HalfCheetahDM-v2 --pert-ratio 1.2 --task mass --seed 1
