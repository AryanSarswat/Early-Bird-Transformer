#!/bin/bash

#SBATCH -JCIFAR10_ViT                             # Job name
#SBATCH -N1 --gres=gpu:RTX_6000:1 --ntasks-per-node=20   # Number of nodes, GPUs, and cores required
#SBATCH -t960                            # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH --mail-type=END,FAIL                  # Mail preferences
#SBATCH --mail-user=asarswat8@gatech.edu           # e-mail address for notifications

cd $HOME/scratch/Early-Bird-Transformer/                         # Change to working directory created in $HOME

module load anaconda3                  # Load module dependencies
conda activate eml
python3 main.py