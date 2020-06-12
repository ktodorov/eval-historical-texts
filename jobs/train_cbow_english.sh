#!/bin/bash
#SBATCH --job-name=cbow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=25:00:00
#SBATCH --mem=60000M
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1

module purge
module load pre2019
module load 2019
module load Miniconda3
module load Python/3.7.5-foss-2019b

source activate eval-env

python run.py --device cuda --seed 13 --eval-freq 2000 --patience 100 --configuration cbow --learning-rate $LR --language $LANGUAGE --corpus $CORPUS --checkpoint-name $CORPUS --challenge semantic-change --batch-size 32 --rnn-hidden-size $HIDDENSIZE --max-training-minutes 1440 --enable-external-logging --skip-validation  --reset-training-on-early-stop --training-reset-epoch-limit 1 --resets-limit 2