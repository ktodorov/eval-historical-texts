#!/bin/bash
#SBATCH --job-name=mult-emb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
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

srun python3 -u run.py --device cuda --seed 13 --configuration multi-fit --learning-rate 1e-2 --metric-types jaccard-similarity levenshtein-distance --language english --challenge ocr --batch-size 8 --hidden-dimension 256 --sentence-piece-vocabulary-size 30522 --embedding-size 32 --dropout 0 --number-of-layers 1 --enable-external-logging --pretrained-weights bert-base-cased --max-training-minutes 4320 --learn-encoder-embeddings > output/multifit-13.txt

cp "$TMPDIR"/eval-historical-texts/wandb $HOME/eval-historical-texts/wandb
cp "$TMPDIR"/eval-historical-texts/results $HOME/eval-historical-texts/results/from-scratch/multifit