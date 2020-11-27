#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0045
#SBATCH --mem 120GB
#SBATCH --gres gpu:1
--comment"#SBATCH --mail-user=sa17826@bristol.ac.uk"
--comment"#SBATCH --mail-type=END,FAIL"

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train_cifar.py
