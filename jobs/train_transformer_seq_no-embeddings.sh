#!/bin/bash
#SBATCH --job-name=seq-no-emb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
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
echo copying finished

cd "$TMPDIR"/eval-historical-texts

srun python3 -u run.py --device cuda --seed 13 --configuration sequence-to-char --learning-rate 1e-3 --metric-types jaccard-similarity levenshtein-distance --language english --challenge post-ocr-correction --batch-size 32 --hidden-dimension 256 --encoder-embedding-size 128 --decoder-embedding-size 16 --dropout 0 --number-of-layers 1 --enable-external-logging --pretrained-weights bert-base-cased --max-training-minutes 4320 --include-pretrained-model --pretrained-model-size 768 --pretrained-max-length 512 --validation-dataset-limit-size 1000 > output/seq-bert-13.txt

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/