#!/bin/bash
#SBATCH --job-name=char-ed-pre
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
    EMBEDDINGSSIZE="32"
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

BATCHSIZEARG="$BATCHSIZE"
if [ -z "$BATCHSIZE" ]
then
    BATCHSIZEARG="64"
fi

EVALFREQARG="$EVALFREQ"
if [ -z "$EVALFREQ" ]
then
    EVALFREQARG="200"
fi

FT=""
FTMODELARG=""
if [ ! -z "$FASTTEXT" ]
then
    FT="-ft"
    FTMODELARG="--fasttext-model $FASTTEXTMODEL"
fi

CHECKPOINTNAME=""
if [ -z "$CHECKPOINT" ]
then
    CHECKPOINTNAME="$LANGUAGE-$PRETR$FT-h$HIDDENSIZE-e$EMBEDDINGSSIZE-l$NUMBERLAYERS$BIARG-d$DR$FTP"
fi

srun python3 -u run.py --device cuda --seed 13 --eval-freq $EVALFREQARG --patience $PATIENCEARG --configuration char-to-char-encoder-decoder --learning-rate $LEARNINGRATE --metric-types jaccard-similarity levenshtein-distance --language $LANGUAGE --challenge post-ocr-correction --batch-size $BATCHSIZEARG --hidden-dimension $HIDDENSIZE --encoder-embedding-size $EMBEDDINGSSIZE --decoder-embedding-size $EMBEDDINGSSIZE --share-embedding-layer --dropout $DROPOUT --number-of-layers $NUMBERLAYERS $BIDIRECTIONAL --enable-external-logging --pretrained-weights $PRETRAINEDWEIGHTS --max-training-minutes 4300 $INCLUDEPRETR --pretrained-model-size 768 --pretrained-max-length 512  --learn-new-embeddings --checkpoint-name $CHECKPOINTNAME $FINETUNEARG $FASTTEXT $FTMODELARG --fasttext-model-size 300

# cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
# cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/
