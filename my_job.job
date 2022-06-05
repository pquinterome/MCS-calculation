#!/bin/bash
#SBATCH -J prostate_calc 
#SBATCH -n 1
#SBATCH -N 4                
# Number of nodes = 4
#SBATCH -o %j.out 
#SBATCH -e %j.err 
#SBATCH -p gpu05,gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

module purge

module load utilities/multi
module load readline/7.0
module load gcc/10.2.0
module load cuda/11.5.0
module load python/anaconda/4.6/miniconda/3.7

echo ">>> Installing Requirements";
conda run -n custom_env pip install -r requirements.txt
echo ">>> Running Code";

/usr/bin/env time -v conda run -n custom_env python code3.py