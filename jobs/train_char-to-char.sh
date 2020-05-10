#!/bin/bash
#SBATCH --job-name=char-pre
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

PRETR=""
if [ ! -z "$INCLUDEPRETR" ]
then
    PRETR="-pretr"
fi

LEARNINGRATE="$LR"
if [ -z "$LR" ]
then
    LEARNINGRATE="1e-3"
fi

HIDDENSIZE="$HIDDEN"
if [ -z "$HIDDEN" ]
then
    HIDDENSIZE="512"
fi

EMBEDDINGSSIZE="$EMB"
if [ -z "$EMB" ]
then
    EMBEDDINGSSIZE="128"
fi

NUMBERLAYERS="$LAYERS"
if [ -z "$LAYERS" ]
then
    NUMBERLAYERS="1"
fi

BIDIRECTIONAL="--bidirectional"
BIARG="-bi"
if [ ! -z "$UNI" ]
then
    BIDIRECTIONAL=""
    BIARG=""
fi

DROPOUT="$DR"
if [ -z "$DR" ]
then
    DROPOUT="0.3"
fi

FINETUNEARG=""
FTP=""
if [ ! -z "$FINETUNE" ]
then
    FINETUNEARG="--fine-tune-pretrained"
    FTP="-tune"
elif [ ! -z "$FINETUNEAFTERCONVERGENCE" ]
then
    FINETUNEARG="--fine-tune-after-convergence"
    FTP="-tune-ac"
fi

if [ ! -z "$FINETUNELR" ]
then
    FINETUNEARG="$FINETUNEARG --fine-tune-learning-rate $FINETUNELR"
    FTP="$FTP$FINETUNELR"
fi

PATIENCEARG="$PATIENCE"
if [ -z "$PATIENCE" ]
then
    PATIENCEARG="10"
fi

CHECKPOINTNAME=""
if [ -z "$CHECKPOINT" ]
then
    CHECKPOINTNAME="english-$PRETR-h$HIDDENSIZE-e$EMBEDDINGSSIZE-l$NUMBERLAYERS$BIARG-d$DR$FTP"
fi

srun python3 -u run.py --device cuda --seed 13 --eval-freq 200 --patience $PATIENCEARG --configuration char-to-char --learning-rate $LEARNINGRATE --metric-types jaccard-similarity levenshtein-distance --language english --challenge post-ocr-correction --batch-size 128 --hidden-dimension $HIDDENSIZE --embeddings-size $EMBEDDINGSSIZE --dropout $DROPOUT --number-of-layers $NUMBERLAYERS --enable-external-logging --pretrained-weights bert-base-cased --max-training-minutes 4320 $INCLUDEPRETR --pretrained-model-size 768 --pretrained-max-length 512  --learn-new-embeddings $BIDIRECTIONAL --checkpoint-name $CHECKPOINTNAME

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/
