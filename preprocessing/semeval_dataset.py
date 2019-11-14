import os
import sys
import pickle

from transformers import *

sys.path.append('..')
from utils import path_utils



def generate_transformer_tokens(corpus_path):
    lines = []
    with open(os.path.join(corpus_path, 'corpus.txt'), 'r', encoding='utf-8') as corpus:
        lines = corpus.readlines()

    pretrained_weights = 'bert-base-cased'
    # pretrained_weights = 'openai-gpt'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    tokenized_text = tokenizer.tokenize(lines[0])
    tokenized_texts = [tokenizer.tokenize(line) for line in lines]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    return input_ids

data_folder = path_utils.combine_path('..', 'data', 'semeval_trial_data')
english_data_folder = path_utils.combine_path(
    data_folder, 'corpora', 'english')

corpus1_path = path_utils.combine_path(
    english_data_folder, 'corpus1')
corpus2_path = path_utils.combine_path(
    english_data_folder, 'corpus2')

ids1 = generate_transformer_tokens(corpus1_path)
ids2 = generate_transformer_tokens(corpus2_path)


ids1_filepath = os.path.join(data_folder, 'ids','english', 'ids1.pickle')
with open(ids1_filepath, 'wb') as handle:
    pickle.dump(ids1, handle, protocol=-1)

ids2_filepath = os.path.join(data_folder, 'ids', 'english', 'ids2.pickle')
with open(ids2_filepath, 'wb') as handle:
    pickle.dump(ids2, handle, protocol=-1)