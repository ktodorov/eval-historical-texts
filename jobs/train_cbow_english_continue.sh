#!/bin/bash
#SBATCH --job-name=cbow-en-1
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

echo copying to SCRATCH...
shopt -s extglob # This is to enable ! on the next line
mkdir -p "$TMPDIR"/eval-historical-texts/results
cp -a $HOME/eval-historical-texts/!(results) "$TMPDIR"/eval-historical-texts
mkdir -p "$TMPDIR"/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE
cp -a $HOME/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE "$TMPDIR"/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE
echo copying finished

cd "$TMPDIR"/eval-historical-texts

srun python3 -u run.py --device cuda --seed 13 --eval-freq 200 --patience 100 --configuration cbow --learning-rate 1e-2 --language $LANGUAGE --corpus $CORPUS --checkpoint-name $CORPUS --challenge semantic-change --batch-size 64 --max-training-minutes 1440 --enable-external-logging --skip-validation  --reset-training-on-early-stop --training-reset-epoch-limit 1 --resets-limit 2 --resume-training --resume-checkpoint-name BEST_$RESUMECORPUS --skip-best-metrics-on-resume

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/