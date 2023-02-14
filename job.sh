#!/bin/bash
​
#SBATCH --job-name=init-methods
#SBATCH --output=/tmp/job-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=4320

JOBDATADIR=`ws create work --space "$SLURM_JOB_ID" --duration "4 00:00:00"`
JOBTMPDIR=/tmp/job-"$SLURM_JOB_ID"

# test for the credentials files
srun test -f ~/CISPA-home/.config/enroot/.credentials
# ​
srun mkdir "$JOBTMPDIR"
# ​
# srun mkdir -p "$JOBDATADIR" "$JOBTMPDIR"/datas

srun --container-image=projects.cispa.saarland:5005#css/ngc/pytorch:22.07-py3 --container-mounts="$JOBTMPDIR":/tmp python3 $HOME/CISPA-scratch/c01gokr/divide_gamma/Thesis_lottery_attractors/main.py
​#
srun mv /tmp/job-"$SLURM_JOB_ID".out "$JOBDATADIR"/out_attractor.txt
srun mv "$JOBTMPDIR"/ "$JOBDATADIR"/data
