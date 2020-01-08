#!/bin/bash
#SBATCH --job-name=train_kbert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
module purge
module load pre2019
module load 2019
# module load eb
module load Miniconda3
module load Python/3.7.5-foss-2018b
# module load cuDNN/7.6.3-CUDA-10.0.130
# module load NCCL/2.4.7-CUDA-10.0.130
# export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

### todo: COPY TO SCRATCH FIRST??? -> see example in vae file
source activate eval-env
srun python3 -u run.py --device cuda --seed 13 --configuration kbert --learning-rate 1e-2 --accuracy-type character-level --language english --challenge sem-eval --batch-size 8 --validation-dataset-reduction-size 0.05 > output/kbert-13.txt