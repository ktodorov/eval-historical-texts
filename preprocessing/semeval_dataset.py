import os
import sys
import pickle

from transformers import PreTrainedTokenizer, BertTokenizer, XLNetTokenizer

sys.path.append('..')
from utils import path_utils



def generate_transformer_tokens(
    corpus_path: str,
    pretrained_weights: str,
    tokenizer: PreTrainedTokenizer):
    lines = []
    with open(os.path.join(corpus_path, 'corpus.txt'), 'r', encoding='utf-8') as corpus:
        lines = corpus.readlines()

    tokenized_text = tokenizer.tokenize(lines[0])
    tokenized_texts = [tokenizer.tokenize(line) for line in lines]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    return input_ids

def preprocess_data(
    corpus_id: int,
    language: str,
    semeval_path: str,
    data_output_path: str,
    pretrained_weights: str,
    tokenizer: PreTrainedTokenizer):
    english_data_folder = path_utils.combine_path(semeval_path, 'corpora', language)
    corpus_path = path_utils.combine_path(english_data_folder, f'corpus{corpus_id}')
    ids = generate_transformer_tokens(corpus_path, pretrained_weights, tokenizer)

    ids_filepath = os.path.join(data_output_path, f'ids{corpus_id}.pickle')
    with open(ids_filepath, 'wb') as handle:
        pickle.dump(ids, handle, protocol=-1)

if __name__ == '__main__':
    semeval_path = path_utils.combine_path('..', 'data', 'semeval_trial_data')
    pretrained_weights = 'bert-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_weights)

    preprocess_data(
        corpus_id=1,
        language='english',
        semeval_path=semeval_path,
        data_output_path=semeval_path,
        pretrained_weights=pretrained_weights,
        tokenizer=tokenizer)

    preprocess_data(
        corpus_id=2,
        language='english',
        semeval_path=semeval_path,
        data_output_path=semeval_path,
        pretrained_weights=pretrained_weights,
        tokenizer=tokenizer)