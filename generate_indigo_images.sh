#!/bin/bash
#SBATCH --job-name=train_ppt
#SBATCH --partition=htc
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/generate_indigo.out
#SBATCH --error=logs/generate_indigo.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate abc_env
nvidia-smi
cd ~/thesis/benchmarking/ABC-Net

echo "Generating..."
python indigo_img_generator.py
echo "Done!"

