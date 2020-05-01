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

FT=""
if [ ! -z "$FASTTEXT" ]
then
    FT="-ft"
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
    ENTITYTAGTYPES="--entity-tag-types literal-fine metonymic-fine component nested "
elif [ $ENTITYTAGS == "coarse" ]
    ENTITYTAGTYPES="--entity-tag-types literal-coarse metonymic-coarse "
fi

srun python3 -u run.py --device cuda --eval-freq 100 --seed 13 --patience 50 --epochs 3000 --configuration rnn-simple --learning-rate 1e-2 --metric-types f1-score precision-score recall-score --language $LANGUAGE --challenge named-entity-recognition --batch-size 128 --enable-external-logging --pretrained-weights $PRETRAINEDWEIGHTS --hidden-dimension 256 --embeddings-size 64 --dropout 0.5 --number-of-layers 1 $ENTITYTAGTYPES --reset-training-on-early-stop --training-reset-epoch-limit 5 $INCLUDEPRETR --pretrained-model-size 768 --pretrained-max-length 512 --learn-new-embeddings --checkpoint-name $LANGUAGE-$LABELTYPE$FT-$PRETRAINEDMODEL$PRETR --no-attention --bidirectional-rnn $FASTTEXT --fasttext-model $FASTTEXTMODEL --fasttext-model-size 300 --max-training-minutes 4300 --merge-subwords --learn-character-embeddings --character-embeddings-size 32 --replace-all-numbers --pretrained-model $PRETRAINEDMODEL

cp -a "$TMPDIR"/eval-historical-texts/wandb/ $HOME/eval-historical-texts/
cp -a "$TMPDIR"/eval-historical-texts/results/ $HOME/eval-historical-texts/