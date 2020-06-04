import os
import sys
import pickle

from typing import List

from enums.language import Language
from services.tokenize.base_tokenize_service import BaseTokenizeService

sys.path.append('..')

def generate_transformer_tokens(
    text_file_paths: List[str],
    tokenize_service: BaseTokenizeService):
    lines = []

    for text_file_path in text_file_paths:
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            lines.extend(text_file.readlines())

    encodings = []
    counter = 0
    step = 500
    while counter < len(lines):
        encodings.extend(tokenize_service.encode_sequences(lines[counter:counter+step]))
        counter += step

    input_ids = [(x[0], x[3]) for x in encodings]
    return input_ids

def preprocess_data(
    corpus_id: int,
    language: Language,
    semeval_path: str,
    data_output_path: str,
    tokenize_service: BaseTokenizeService):
    language_data_folder = path_utils.combine_path(semeval_path, str(language))
    corpus_path = path_utils.combine_path(language_data_folder, f'corpus{corpus_id}', 'lemma')

    text_file_paths = []
    for file_name in os.listdir(corpus_path):
        if file_name.endswith('.txt'):
            text_file_paths.append(os.path.join(corpus_path, file_name))

    ids = generate_transformer_tokens(text_file_paths, tokenize_service)

    ids_filepath = os.path.join(data_output_path, f'ids{corpus_id}.pickle')
    with open(ids_filepath, 'wb') as handle:
        pickle.dump(ids, handle, protocol=-1)