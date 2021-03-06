#!/bin/bash
#SBATCH --job-name=kbert-en-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=73:00:00
#SBATCH --mem=60000M
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:1

module purge
module load pre2019
module load 2019
module load Miniconda3
module load Python/3.7.5-foss-2019b

LANGUAGE='english'
CORPUS=1

source activate eval-env

echo copying to SCRATCH...
shopt -s extglob # This is to enable ! on the next line
mkdir -p "$TMPDIR"/eval-historical-texts/results
cp -a $HOME/eval-historical-texts/!(results) "$TMPDIR"/eval-historical-texts
mkdir -p "$TMPDIR"/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE
cp -a $HOME/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE "$TMPDIR"/eval-historical-texts/results/semantic-change/kbert/$LANGUAGE
echo copying finished

cd "$TMPDIR"/eval-historical-texts

srun python3 -u run.py --device cuda --seed 13 --eval-freq 600 --patience 100 --configuration kbert --learning-rate 1e-5 --language $LANGUAGE --corpus $CORPUS --challenge semantic-change --batch-size 4 --pretrained-weights bert-base-cased --pretrained-max-length 512 --max-training-minutes 4320 --enable-external-logging --skip-validation --checkpoint-name $CORPUS --reset-training-on-early-stop --training-reset-epoch-limit 1 --resets-limit 2 > output/semeval-$LANGUAGE-$CORPUS-13.txt

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/