[< go back to main page](../../README.md)

# Post-OCR correction arguments

## Description

This page describes the arguments used in the `post-ocr-correction` challenge and `char-to-char-encoder-decoder` configuration. These derive from the [PostOCRArgumentsService](../../services/arguments/postocr_arguments_service.py).

## Model

We combine the embedding layer with an encoder-decoder architecture with attention (similar to [this one](https://arxiv.org/abs/1409.0473)). Thus we consider the raw OCR version of a text and ground truth as two distinct languages. We then train the model to translate from the first to the latter. Our architecture works at a character level and thus also concatenates pre-trained sub-word information to every character. This is visible in the first figure below. The second figure shows an example pass through the decoder.

<table>
    <tr>
        <td style='border-right: 1px solid black;'>
            <img src="../images/subword-to-char.png">
        </td>
        <td style='border-left: 1px solid black;'>
            <img src="../images/decoder-attention-pass.png">
        </td>
    </tr>
</table>

## Arguments

| Argument     | Type          | Default value  | Description |
| ------------- | ------------- | -------------- |-------------|
| `encoder-embedding-size` | `int` | 128 | The size used for generating embeddings in the encoder |
| `decoder-embedding-size` | `int` | 16 | The size used for generating embeddings in the decoder |
| `share-embedding-layer` | `bool` | False | If set to true, the embedding layer of the encoder and decoder will be shared |
| `hidden-dimension` | `int` | 256 | The dimension size used for hidden layers |
| `dropout` | `float` | 0.0 | Dropout probability |
| `number-of-layers` | `int` | 1 | Number of layers used for RNN or Transformer models |
| `bidirectional` | `bool` | False | Whether the RNN used will be bidirectional |
| `use-beam-search` | `bool` | False | If set to true, beam search will be used for decoding instead of greedy decoding |
| `beam-width` | `int` | 3 | Width of the beam when using beam search. Defaults to 3 |


## Usage

Example of usage cases for the Post-OCR correction challenge

### Training

```bash
python run.py --challenge post-ocr-correction --configuration char-to-char-encoder-decoder --device cuda --eval-freq 50 --seed 13 --learning-rate 1e-3 --metric-types levenshtein-distance jaccard-similarity --language english --checkpoint-name english-post-ocr --batch-size 4 --pretrained-weights bert-base-cased --pretrained-model-size 768 --pretrained-max-length 512 --include-fasttext-model --fasttext-model en-ft.bin --fasttext-model-size 300 --learn-new-embeddings --share-embedding-layer --hidden-dimension 512 --encoder-embedding-size 64 --decoder-embedding-size 64 --dropout 0.5 --number-of-layers 2 --bidirectional --patience 10000
```

### Evaluation

```bash
python run.py --challenge post-ocr-correction --configuration char-to-char-encoder-decoder --device cuda --seed 13 --language english --batch-size 32 --checkpoint-name english-post-ocr --evaluate --evaluation-type jaccard-similarity levenshtein-edit-distance-improvement --pretrained-weights bert-base-cased --include-pretrained-model --fine-tune-pretrained --fine-tune-learning-rate 1e-4 --pretrained-model-size 768 --pretrained-max-length 512 --include-fasttext-model --fasttext-model en-ft.bin --fasttext-model-size 300 --learn-new-embeddings --share-embedding-layer --hidden-dimension 512 --encoder-embedding-size 64 --decoder-embedding-size 64 --dropout 0.5 --number-of-layers 2 --bidirectional
```

## Data

The data which is provided by the organizers of the ICDAR 2019 challenge includes noisy OCR of printed texts from different sources and ten languages. We focus on English, German and French. We additionally make use of the full French and English data from the ICDAR 2017 challenge. Finally we use Overproof data in English ([link to website](https://overproof.projectcomputing.com/)) to further increase our dataset size.