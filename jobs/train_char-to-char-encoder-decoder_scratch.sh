#!/bin/bash
#SBATCH --job-name=char-ed-scr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:05:00
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

srun python3 -u run.py --device cuda --seed 13 --eval-freq 200 --configuration char-to-char-encoder-decoder --learning-rate 1e-3 --metric-types jaccard-similarity levenshtein-distance --language english --challenge post-ocr-correction --batch-size 128 --hidden-dimension 512 --encoder-embedding-size 32 --decoder-embedding-size 32 --share-embedding-layer --dropout 0.2 --number-of-layers 1 --bidirectional --enable-external-logging --max-training-minutes 4300 --learn-new-embeddings --use-beam-search --checkpoint-name scratch --validation-dataset-limit-size 1024 > output/char-to-char-enc-dec-scratch-13.txt

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/
