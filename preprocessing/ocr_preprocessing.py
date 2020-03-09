import os
import urllib.request
import random
from shutil import copyfile
from multiprocessing import Pool, TimeoutError
import functools
import sys
import pickle

from typing import List, Tuple

from transformers import PreTrainedTokenizer

from entities.language_data import LanguageData
from utils import path_utils

from services.vocabulary_service import VocabularyService
from services.tokenizer_service import TokenizerService
from services.metrics_service import MetricsService


def preprocess_data(
        tokenizer_service: TokenizerService,
        metrics_service: MetricsService,
        vocabulary_service: VocabularyService,
        pickles_path: str,
        full_data_path: str,
        data_output_path: str):

    train_language_data, validation_language_data = parse_language_data(
        tokenizer_service,
        metrics_service,
        vocabulary_service,
        pickles_path,
        full_data_path)

    train_language_data_filepath = os.path.join(
        data_output_path, f'train_language_data.pickle')
    validation_language_data_filepath = os.path.join(
        data_output_path, f'validation_language_data.pickle')

    with open(train_language_data_filepath, 'wb') as train_handle:
        pickle.dump(train_language_data, train_handle, protocol=-1)

    with open(validation_language_data_filepath, 'wb') as validation_handle:
        pickle.dump(validation_language_data, validation_handle, protocol=-1)


def cut_string(text: str, chunk_length: int):
    result = [text[i:i+chunk_length]
              for i in range(0, len(text), chunk_length)]
    return result


def read_ocr_file(file_path: str, start_position: int) -> List[Tuple[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as language_file:
        text_data: List[str] = language_file.read().split('\n')

        string_length = 50
        ocr_strings = cut_string(text_data[1][start_position:], string_length)
        gs_strings = cut_string(text_data[2][start_position:], string_length)

        if len(ocr_strings) != len(gs_strings):
            return None

        return [(ocr_strings[i], gs_strings[i]) for i in range(len(ocr_strings))]


def save_data_files(full_data_path: str, full_ocr_path: str, full_gs_path: str):
    ocr_aligned_lengths = []
    gs_aligned_lengths = []
    file_paths = []

    for i, file_name in enumerate(os.listdir(full_data_path)):
        file_paths.append(os.path.join(full_data_path, file_name))

    number_of_files = len(file_paths)
    file_data = []
    for i, file_path in enumerate(file_paths):
        print(f'{i}/{number_of_files}             \r', end='')
        result = read_ocr_file(file_path, start_position=14)
        if result is not None:
            file_data.extend(result)

    ocr_file_data = [x[0] for x in file_data]
    gs_file_data = [x[1] for x in file_data]

    with open(full_ocr_path, 'wb') as ocr_handle:
        pickle.dump(ocr_file_data, ocr_handle, protocol=-1)

    with open(full_gs_path, 'wb') as gs_handle:
        pickle.dump(gs_file_data, gs_handle, protocol=-1)

    return ocr_file_data, gs_file_data


def read_data(
        full_data_path: str,
        full_ocr_path: str,
        full_gs_path: str,
        full_ocr_tokens_path: str,
        full_gs_tokens_path: str,
        tokenizer_service: TokenizerService):

    if not os.path.exists(full_ocr_path) or not os.path.exists(full_gs_path):
        ocr_file_data, gs_file_data = save_data_files(
            full_data_path,
            full_ocr_path,
            full_gs_path)
    else:
        with open(full_ocr_path, 'rb') as ocr_handle:
            ocr_file_data = pickle.load(ocr_handle)

        with open(full_gs_path, 'rb') as gs_handle:
            gs_file_data = pickle.load(gs_handle)

    if not os.path.exists(full_ocr_tokens_path) or not os.path.exists(full_gs_tokens_path):
        ocr_tokens = []
        gs_tokens = []
        skipped_indices = []
        for i in range(len(ocr_file_data)):
            current_ids, _, _, _ = tokenizer_service.encode_sequence(
                ocr_file_data[i])
            if len(current_ids) > 2000:
                skipped_indices.append(i)
                continue

            gs_ids, _, _, _ = tokenizer_service.encode_sequence(
                gs_file_data[i])

            ocr_tokens.append(current_ids)
            gs_tokens.append(gs_ids)

        with open(full_ocr_tokens_path, 'wb') as ocr_handle:
            pickle.dump(ocr_tokens, ocr_handle, protocol=-1)

        with open(full_gs_tokens_path, 'wb') as gs_handle:
            pickle.dump(gs_tokens, gs_handle, protocol=-1)

        for index in sorted(skipped_indices, reverse=True):
            del ocr_file_data[index]
            del gs_file_data[index]

        with open(full_ocr_path, 'wb') as ocr_handle:
            pickle.dump(ocr_file_data, ocr_handle, protocol=-1)

        with open(full_gs_path, 'wb') as gs_handle:
            pickle.dump(gs_file_data, gs_handle, protocol=-1)
    else:
        with open(full_ocr_tokens_path, 'rb') as ocr_handle:
            ocr_tokens = pickle.load(ocr_handle)

        with open(full_gs_tokens_path, 'rb') as gs_handle:
            gs_tokens = pickle.load(gs_handle)

    return ocr_file_data, gs_file_data, ocr_tokens, gs_tokens


def save_metrics_obj(
        token_pairs,
        decoded_pairs,
        jaccard_similarities,
        levenshtein_distances,
        pickles_path: str):
    metrics_path = os.path.join(pickles_path, 'metrics.pickle')

    metrics_obj = {
        'token_pairs': token_pairs,
        'decoded_pairs': decoded_pairs,
        'jaccard_similarities': jaccard_similarities,
        'levenshtein_distances': levenshtein_distances,
    }

    with open(metrics_path, 'wb') as metrics_handle:
        pickle.dump(metrics_obj, metrics_handle, protocol=-1)

    print('Saved metrics')


def load_metrics_obj(pickles_path: str):
    metrics_path = os.path.join(pickles_path, 'metrics.pickle')
    if not os.path.exists(metrics_path):
        return (None, None, None, None)

    with open(metrics_path, 'rb') as metrics_handle:
        metrics_obj = pickle.load(metrics_handle)

    return (metrics_obj['token_pairs'],
            metrics_obj['decoded_pairs'],
            metrics_obj['jaccard_similarities'],
            metrics_obj['levenshtein_distances'])


def parse_metrics_obj(
        tokenizer_service: TokenizerService,
        metrics_service: MetricsService,
        ocr_tokens: List[List[int]],
        gs_tokens: List[List[int]],
        ocr_file_data: List[str],
        gs_file_data: List[str],
        pickles_path: str):
    token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances = load_metrics_obj(pickles_path)

    if token_pairs is None:
        token_pairs = [([tokenizer_service.id_to_token(x) for x in ocr_tokens[i]], [
                        tokenizer_service.id_to_token(x) for x in gs_tokens[i]]) for i in range(len(ocr_tokens))]
        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    if decoded_pairs is None:
        decoded_pairs = [(ocr_file_data[i], gs_file_data[i])
                         for i in range(len(ocr_tokens))]
        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    all_pairs = len(token_pairs)
    # if jaccard_similarities is None:
    #     jaccard_similarities = []
    #     for i, token_pair in enumerate(token_pairs):
    #         jaccard_similarities.append(
    #             metrics_service.calculate_jaccard_similarity(token_pair[0], token_pair[1]))

    #     save_metrics_obj(token_pairs, decoded_pairs,
    #                      jaccard_similarities, levenshtein_distances, pickles_path)

    # if levenshtein_distances is None:
    #     levenshtein_distances = []

    # if len(levenshtein_distances) < all_pairs:
    #     for i, decoded_pair in enumerate(decoded_pairs):
    #         if i < len(levenshtein_distances):
    #             continue

    #         print(f'LEVENSHTEIN - {i}/{all_pairs}             \r', end='')
    #         levenshtein_distances.append(
    #             metrics_service.calculate_normalized_levenshtein_distance(decoded_pair[0], decoded_pair[1]))

    #         if i % 50000 == 0:
    #             save_metrics_obj(token_pairs, decoded_pairs,
    #                              jaccard_similarities, levenshtein_distances, pickles_path)

    #     save_metrics_obj(token_pairs, decoded_pairs,
    #                      jaccard_similarities, levenshtein_distances, pickles_path)

    return token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances


def load_split_data(
        tokenizer_service: TokenizerService,
        metrics_service: MetricsService,
        full_data_path: str,
        full_ocr_path: str,
        full_gs_path: str,
        full_ocr_tokens_path: str,
        full_gs_tokens_path: str,
        pickles_path: str,
        train_pickle_path: str,
        validation_pickle_path: str):

    if not os.path.exists(train_pickle_path) or not os.path.exists(validation_pickle_path):
        ocr_file_data, gs_file_data, ocr_tokens, gs_tokens = read_data(
            full_data_path,
            full_ocr_path,
            full_gs_path,
            full_ocr_tokens_path,
            full_gs_tokens_path,
            tokenizer_service)

        token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances = parse_metrics_obj(
            tokenizer_service,
            metrics_service,
            ocr_tokens,
            gs_tokens,
            ocr_file_data,
            gs_file_data,
            pickles_path)

        eval_indices = random.sample(
            range(len(token_pairs)), int(0.2 * len(token_pairs)))

        train_pairs = []
        eval_pairs = [ [token_pairs[i], decoded_pairs[i]] for i in eval_indices ]

        eval_indices_dict = { i: False for i in range(len(token_pairs)) }
        for i in eval_indices:
            eval_indices_dict[i] = True

        train_pairs = [ [token_pairs[i], decoded_pairs[i]] for i in range(len(token_pairs)) if not eval_indices_dict[i]]
        # for i in range(len(token_pairs)):
        #     if i in eval_indices:
        #         eval_pairs.append([token_pairs[i], decoded_pairs[i]])
        #     else:
        #         train_pairs.append([token_pairs[i], decoded_pairs[i]])

        with open(train_pickle_path, 'wb') as train_handle:
            pickle.dump(train_pairs, train_handle, protocol=-1)

        with open(validation_pickle_path, 'wb') as eval_handle:
            pickle.dump(eval_pairs, eval_handle, protocol=-1)
    else:
        with open(train_pickle_path, 'rb') as train_pickle_file:
            train_pairs = pickle.load(train_pickle_file)

        with open(validation_pickle_path, 'rb') as validation_pickle_file:
            eval_pairs = pickle.load(validation_pickle_file)

    return train_pairs, eval_pairs


def parse_language_data(
        tokenizer_service: TokenizerService,
        metrics_service: MetricsService,
        vocabulary_service: VocabularyService,
        pickles_path: str,
        full_data_path: str) -> LanguageData:

    train_pickle_path = os.path.join(pickles_path, 'train_pairs.pickle')
    validation_pickle_path = os.path.join(pickles_path, 'eval_pairs.pickle')

    full_ocr_path = os.path.join(pickles_path, 'combined_ocr.pickle')
    full_gs_path = os.path.join(pickles_path, 'combined_gs.pickle')

    full_ocr_tokens_path = os.path.join(
        pickles_path, 'combined_ocr_tokens.pickle')
    full_gs_tokens_path = os.path.join(
        pickles_path, 'combined_gs_tokens.pickle')

    train_pairs, validation_pairs = load_split_data(
        tokenizer_service,
        metrics_service,
        full_data_path,
        full_ocr_path,
        full_gs_path,
        full_ocr_tokens_path,
        full_gs_tokens_path,
        pickles_path,
        train_pickle_path,
        validation_pickle_path)

    train_language_data = LanguageData([], [], [], [], [])
    for train_pair in train_pairs:
        train_language_data.add_entry(
            None, train_pair[0][0], train_pair[0][1], train_pair[1][0], train_pair[1][1], tokenizer_service, vocabulary_service)

    validation_language_data = LanguageData([], [], [], [], [])
    for validation_pair in validation_pairs:
        validation_language_data.add_entry(
            None, validation_pair[0][0], validation_pair[0][1], validation_pair[1][0], validation_pair[1][1], tokenizer_service, vocabulary_service)

    return train_language_data, validation_language_data
