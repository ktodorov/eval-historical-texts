[< go back to main page](../../README.md)

# Named entity recognition arguments

## Description

This section describes the arguments used in the `named-entity-recognition` challenge and `bi-lstm-crf` configuration. These derive from the [NERArgumentsService](../../services/arguments/ner_arguments_service.py).

## Arguments

| Parameter     | Type          | Default value  | Description |
| ------------- | ------------- | -------------- |-------------|
| `embeddings-size` | `int` | 128 | The size used for generating sub-word embeddings |
| `hidden-dimension` | `int` | 256 | The dimension size used for hidden layers |
| `dropout` | `float` | 0.0 | Dropout probability |
| `number-of-layers` | `int` | 1 | Number of layers used for the RNN |
| `entity-tag-types` | [`EntityTagType`](../../../enums/entity_tag_type.py) | LiteralCoarse | Entity tag types that will be used for classification. Default is only to use Literal coarse
| `no-attention` | `bool` | False | Whether to skip the attention layer |
| `bidirectional-rnn` | `bool` | False | Whether to use a bidirectional version of the RNN |
| `merge-subwords` | `bool` | False | Whether to merge the sub-word embeddings before passing through the RNN |
| `learn-character-embeddings` | `bool` | False | Whether to learn character embeddings next to the default sub-word ones | |
| `character-embeddings-size` | `int` | None | The size used for generating character embeddings |
| `character-hidden-size` | `int` | None | The hidden size used for the character embeddings RNN
| `replace-all-numbers` | `bool` | False | If all numbers should be replaced by a hard-coded fixed string |
| `use-weighted-loss` | `bool` | False | If set to true, CRF layer will use weighted loss which focuses more on non-empty tags |
| `use-manual-features` | `bool` | False | If set to true, manual features representations will be learned and added to general embeddings | |
| `split-type` | [`TextSequenceSplitType`](../../enums/text_sequence_split_type.py) | Documents | This sets the split level of the input data. Default is document-level

## Usage

Example of usage cases for the Named Entity Recognition challenge

### Training

```bash
python run.py --challenge named-entity-recognition --configuration bi-lstm-crf --epochs 100000 --device cuda --eval-freq 5 --seed 13 --learning-rate 1e-2 --metric-types f1-score precision recall --language english --batch-size 32 --checkpoint-name english-ner --no-attention --pretrained-weights bert-base-cased --pretrained-model-size 768 --pretrained-max-length 512 --learn-new-embeddings --hidden-dimension 128 --bidirectional-rnn --number-of-layers 1 --embeddings-size 32 --learn-character-embeddings --character-embeddings-size 16 --character-hidden-size 32 --replace-all-numbers --merge-subwords --split-type segment --entity-tag-types literal-fine literal-coarse metonymic-fine metonymic-coarse component nested --reset-training-on-early-stop --training-reset-epoch-limit 5 --patience 3
```

### Evaluation

```bash
python run.py --challenge named-entity-recognition --configuration bi-lstm-crf --device cuda --seed 13 --metric-types f1-score --language german  --checkpoint-name german-ner --batch-size 1 --evaluate --pretrained-weights bert-base-german-cased --pretrained-model bert --include-pretrained-model --pretrained-model-size 768 --pretrained-max-length 512 --include-fasttext-model --fasttext-model de-ft-model.bin --fasttext-model-size 300 --hidden-dimension 512 --embeddings-size 128 --number-of-layers 1 --dropout 0.5 --bidirectional-rnn --no-attention --learn-character-embeddings --character-embeddings-size 16 --character-hidden-size 32 --replace-all-numbers --merge-subwords --split-type document --entity-tag-types literal-fine
```