#!/bin/bash
#SBATCH --job-name=ner-pre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=25:00:00
#SBATCH --mem=60000M
#SBATCH -p gpu_shared

module purge
module load pre2019
module load 2019
module load Miniconda3
module load Python/3.7.5-foss-2019b

source activate eval-env

# echo copying to SCRATCH...
# shopt -s extglob # This is to enable ! on the next line
# mkdir -p "$TMPDIR"/eval-historical-texts/results
# cp -a $HOME/eval-historical-texts/!(results) "$TMPDIR"/eval-historical-texts
# echo copying finished

# cd "$TMPDIR"/eval-historical-texts

FT=""
FTMODELARG=""
if [ ! -z "$FASTTEXT" ]
then
    FT="-ft"
    FTMODELARG="--fasttext-model $FASTTEXTMODEL"
fi


PRETR=""
if [ ! -z "$INCLUDEPRETR" ]
then
    PRETR="-pretr"
fi

if [ -z "$PRETRAINEDMODEL" ]
then
    PRETRAINEDMODEL="bert"
fi

ENTITYTAGTYPES=""
if [ $ENTITYTAGS == "all" ]
then
    ENTITYTAGTYPES="--entity-tag-types literal-fine literal-coarse metonymic-fine metonymic-coarse component nested "
elif [ $ENTITYTAGS == "fine" ]
then
    ENTITYTAGTYPES="--entity-tag-types literal-fine metonymic-fine component nested "
elif [ $ENTITYTAGS == "coarse" ]
then
    ENTITYTAGTYPES="--entity-tag-types literal-coarse metonymic-coarse "
elif [ $ENTITYTAGS == "1" ]
then
    ENTITYTAGTYPES="--entity-tag-types literal-fine "
elif [ $ENTITYTAGS == "2" ]
then
    ENTITYTAGTYPES="--entity-tag-types literal-coarse "
elif [ $ENTITYTAGS == "3" ]
then
    ENTITYTAGTYPES="--entity-tag-types metonymic-fine "
elif [ $ENTITYTAGS == "4" ]
then
    ENTITYTAGTYPES="--entity-tag-types metonymic-coarse "
elif [ $ENTITYTAGS == "5" ]
then
    ENTITYTAGTYPES="--entity-tag-types component "
elif [ $ENTITYTAGS == "6" ]
then
    ENTITYTAGTYPES="--entity-tag-types nested "
fi

CHARACTEREMBEDDINGS=""
CHARCHECKPOINT=""
if [ ! -z "$CHARESIZE" ]
then
    CHARACTEREMBEDDINGS="--learn-character-embeddings --character-embeddings-size $CHARESIZE --character-hidden-size $CHARHSIZE"
    CHARCHECKPOINT="-ce$CHARESIZE-ch$CHARHSIZE"
fi

LEARNINGRATE="$LR"
if [ -z "$LR" ]
then
    LEARNINGRATE="1e-3"
fi

HIDDENSIZE="$HIDDEN"
if [ -z "$HIDDEN" ]
then
    HIDDENSIZE="256"
fi

EMBEDDINGSSIZE="$EMB"
if [ -z "$EMB" ]
then
    EMBEDDINGSSIZE="64"
fi

NUMBERLAYERS="$LAYERS"
if [ -z "$LAYERS" ]
then
    NUMBERLAYERS="1"
fi

BIDIRECTIONAL="--bidirectional-rnn"
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

WEIGHTEDLOSSARG=""
WL=""
if [ ! -z "$WEIGHTED" ]
then
    WEIGHTEDLOSSARG="--use-weighted-loss"
    WL="-wl"
fi

MANUALFEATURESARG=""
MANFEAT=""
if [ ! -z "$MANUALFEATURES" ]
then
    MANUALFEATURESARG="--use-manual-features"
    MANFEAT="-mf"
fi

NEWEMBEDDINGSARG="--learn-new-embeddings"
NEWEMB=""
if [ ! -z "$NONEWEMBEDDINGS" ]
then
    NEWEMBEDDINGSARG=""
    NEWEMB="-nonew"
fi

SPLITARG="--split-type document"
SPL="-spl-d"
if [ $SPLIT == "seg" ]
then
    SPLITARG="--split-type segment"
    SPL="-spl-s"
elif [ $SPLIT == "multi-seg" ]
then
    SPLITARG="--split-type multi-segment"
    SPL="-spl-ms"
fi

PATIENCEARG="$PATIENCE"
if [ -z "$PATIENCE" ]
then
    PATIENCEARG="10"
fi

SEEDARG="$SEED"
if [ -z "$SEED" ]
then
    SEEDARG="13"
fi

CHECKPOINTNAME=""
if [ -z "$CHECKPOINT" ]
then
    CHECKPOINTNAME="$LANGUAGE-$ENTITYTAGS-$FT-$PRETRAINEDMODEL$PRETR$CHARCHECKPOINT-h$HIDDENSIZE-e$EMBEDDINGSSIZE-l$NUMBERLAYERS$BIARG-d$DR$FTP$WL$MANFEAT$NEWEMB$SPL-seed$SEEDARG"
fi

EVALFREQ="$EF"
if [ -z "$EF" ]
then
    EVALFREQ="300"
fi

BATCHSIZE="$BATCH"
if [ -z "$BATCH" ]
then
    BATCHSIZE="128"
fi

python run.py --device cuda --eval-freq $EVALFREQ --seed $SEEDARG --patience $PATIENCEARG --epochs 500 --configuration bi-lstm-crf --learning-rate $LEARNINGRATE --metric-types f1-score precision recall --language $LANGUAGE --challenge named-entity-recognition --batch-size $BATCHSIZE --enable-external-logging --pretrained-weights $PRETRAINEDWEIGHTS --hidden-dimension $HIDDENSIZE --embeddings-size $EMBEDDINGSSIZE --dropout $DROPOUT --number-of-layers $NUMBERLAYERS $ENTITYTAGTYPES --reset-training-on-early-stop --training-reset-epoch-limit 5 $INCLUDEPRETR --pretrained-model-size 768 --pretrained-max-length 512 $NEWEMBEDDINGSARG --checkpoint-name $CHECKPOINTNAME --no-attention $BIDIRECTIONAL $FASTTEXT $FTMODELARG --fasttext-model-size 300 --max-training-minutes 1440 --merge-subwords $CHARACTEREMBEDDINGS --replace-all-numbers --pretrained-model $PRETRAINEDMODEL $FINETUNEARG $WEIGHTEDLOSSARG $MANUALFEATURESARG $SPLITARG

# cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
# cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/