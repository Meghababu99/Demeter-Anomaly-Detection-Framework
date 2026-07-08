#!/bin/bash
#SBATCH --job-name=hp-B
#SBATCH --output=Out-hp-B.txt
#SBATCH --error=Err-hp-B.txt
#SBATCH --partition=gpu-V100
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH --cores-per-socket=2
#SBATCH --nice=0

cd ~/GRID-METHODS/LSTM-AE-HO/Demeter/Experiments/
source NN/bin/activate
python Retraining_Hp-B.py
