#!/bin/bash
#SBATCH --job-name=seq-emb
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
module load Python/3.7.5-foss-2018b

source activate eval-env

echo copying to SCRATCH...
shopt -s extglob # This is to enable ! on the next line
mkdir -p "$TMPDIR"/eval-historical-texts/results
cp -a $HOME/eval-historical-texts/!(results) "$TMPDIR"/eval-historical-texts
echo copying finished

cd "$TMPDIR"/eval-historical-texts

srun python3 -u run.py --device cuda --seed 13 --eval-freq 200 --configuration sequence-to-char --learning-rate 1e-4 --metric-types jaccard-similarity levenshtein-distance --language english --challenge post-ocr-correction --batch-size 256 --hidden-dimension 256 --pretrained-vocabulary-size 30522 --encoder-embedding-size 64 --decoder-embedding-size 64 --dropout 0.2 --number-of-layers 1 --enable-external-logging --pretrained-weights bert-base-cased --max-training-minutes 4320 --learn-new-embeddings --validation-dataset-limit-size 50000 > output/seq-scratch-13.txt

cp "$TMPDIR"/eval-historical-texts/wandb $HOME/eval-historical-texts/wandb
cp "$TMPDIR"/eval-historical-texts/results $HOME/eval-historical-texts/results/from-scratch/seq-to-char
