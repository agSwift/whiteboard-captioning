#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ag2719
#SBATCH --output=/vol/bitbucket/ag2719/drawing-gui/slurm_outputs/slurm_%j.out
export PATH=/vol/bitbucket/ag2719/drawing/bin/:$PATH
source activate

. /vol/cuda/12.0.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/ag2719/drawing-gui/ && srun python ./iam/train.py

