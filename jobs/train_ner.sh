#!/bin/bash
#SBATCH --job-name=ner-pre
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
module load Python/3.7.5-foss-2018b

source activate eval-env

echo copying to SCRATCH...
shopt -s extglob # This is to enable ! on the next line
mkdir -p "$TMPDIR"/eval-historical-texts/results
cp -a $HOME/eval-historical-texts/!(results) "$TMPDIR"/eval-historical-texts
echo copying finished

cd "$TMPDIR"/eval-historical-texts

LABELTYPE='coarse'
LANGUAGE='french'

srun python3 -u run.py --device cuda --eval-freq 100 --seed 13 --configuration rnn-simple --learning-rate 1e-4 --metric-types f1-score --language $LANGUAGE --challenge named-entity-recognition --batch-size 128 --enable-external-logging --pretrained-weights bert-base-multilingual-cased --hidden-dimension 256 --embeddings-size 64 --label-type $LABELTYPE --reset-training-on-early-stop --training-reset-epoch-limit 5 --include-pretrained-model --pretrained-model-size 768 --pretrained-max-length 512 --learn-new-embeddings --checkpoint-name $LANGUAGE-$LABELTYPE-pretr > output/ner-bert-$LANGUAGE-$LABELTYPE-13.txt

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/