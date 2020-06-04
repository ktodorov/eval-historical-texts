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
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.metrics_service import MetricsService
from services.data_service import DataService


def preprocess_data(
        tokenize_service: BaseTokenizeService,
        metrics_service: MetricsService,
        vocabulary_service: VocabularyService,
        data_service: DataService,
        pickles_path: str,
        full_data_path: str,
        data_output_path: str,
        split_data: bool = True):

    language_data_result = parse_language_data(
        tokenize_service,
        metrics_service,
        vocabulary_service,
        data_service,
        pickles_path,
        full_data_path,
        split_data=split_data)

    if split_data:
        train_language_data, validation_language_data = language_data_result
        train_language_data_filepath = os.path.join(
            data_output_path, f'train_language_data.pickle')
        validation_language_data_filepath = os.path.join(
            data_output_path, f'validation_language_data.pickle')

        with open(train_language_data_filepath, 'wb') as train_handle:
            pickle.dump(train_language_data, train_handle, protocol=-1)

        with open(validation_language_data_filepath, 'wb') as validation_handle:
            pickle.dump(validation_language_data, validation_handle, protocol=-1)
    else:
        test_language_data_filepath = os.path.join(
            data_output_path, f'test_language_data.pickle')
        with open(test_language_data_filepath, 'wb') as test_handle:
            pickle.dump(language_data_result, test_handle, protocol=-1)


def save_data_files(
        data_service: DataService,
        full_data_path: str,
        full_ocr_path: str,
        full_gs_path: str):
    ocr_aligned_lengths = []
    gs_aligned_lengths = []
    file_paths = []
    file_names = os.listdir(full_data_path)
    number_of_files = len(file_names)

    ocr_file_data = []
    gs_file_data = []

    for i, file_name in enumerate(file_names):
        print(f'{i}/{number_of_files}             \r', end='')
        result = data_service.load_python_obj(
            full_data_path, file_name, extension_included=True)

        ocr_file_data.extend(result[0])
        gs_file_data.extend(result[1])

    with open(full_ocr_path, 'wb') as ocr_handle:
        pickle.dump(ocr_file_data, ocr_handle, protocol=-1)

    with open(full_gs_path, 'wb') as gs_handle:
        pickle.dump(gs_file_data, gs_handle, protocol=-1)

    return ocr_file_data, gs_file_data


def read_data(
        tokenize_service: BaseTokenizeService,
        data_service: DataService,
        full_data_path: str,
        full_ocr_path: str,
        full_gs_path: str,
        full_ocr_tokens_path: str,
        full_gs_tokens_path: str):

    if not os.path.exists(full_ocr_path) or not os.path.exists(full_gs_path):
        ocr_file_data, gs_file_data = save_data_files(
            data_service,
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
            current_ids, _, _, _ = tokenize_service.encode_sequence(
                ocr_file_data[i])
            if len(current_ids) > 2000:
                skipped_indices.append(i)
                continue

            gs_ids, _, _, _ = tokenize_service.encode_sequence(
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
        tokenize_service: BaseTokenizeService,
        metrics_service: MetricsService,
        ocr_tokens: List[List[int]],
        gs_tokens: List[List[int]],
        ocr_file_data: List[str],
        gs_file_data: List[str],
        pickles_path: str):
    token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances = load_metrics_obj(
        pickles_path)

    if token_pairs is None:
        token_pairs = [([tokenize_service.id_to_token(x) for x in ocr_tokens[i]], [
                        tokenize_service.id_to_token(x) for x in gs_tokens[i]]) for i in range(len(ocr_tokens))]
        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    if decoded_pairs is None:
        decoded_pairs = [(ocr_file_data[i], gs_file_data[i])
                         for i in range(len(ocr_tokens))]
        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    all_pairs = len(token_pairs)
    if jaccard_similarities is None:
        jaccard_similarities = []
        for i, token_pair in enumerate(token_pairs):
            jaccard_similarities.append(
                metrics_service.calculate_jaccard_similarity(token_pair[0], token_pair[1]))

        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    if levenshtein_distances is None:
        levenshtein_distances = []

    if len(levenshtein_distances) < all_pairs:
        for i, decoded_pair in enumerate(decoded_pairs):
            if i < len(levenshtein_distances):
                continue

            print(f'LEVENSHTEIN - {i}/{all_pairs}             \r', end='')
            levenshtein_distances.append(
                metrics_service.calculate_normalized_levenshtein_distance(decoded_pair[0], decoded_pair[1]))

            if i % 150000 == 0:
                save_metrics_obj(token_pairs, decoded_pairs,
                                 jaccard_similarities, levenshtein_distances, pickles_path)

        save_metrics_obj(token_pairs, decoded_pairs,
                         jaccard_similarities, levenshtein_distances, pickles_path)

    return token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances


def load_split_data(
        tokenize_service: BaseTokenizeService,
        metrics_service: MetricsService,
        data_service: DataService,
        full_data_path: str,
        full_ocr_path: str,
        full_gs_path: str,
        full_ocr_tokens_path: str,
        full_gs_tokens_path: str,
        pickles_path: str,
        train_pickle_path: str = None,
        validation_pickle_path: str = None,
        test_pickle_path: str = None):

    assert (train_pickle_path is not None and validation_pickle_path is not None) or test_pickle_path is not None, 'Either train/validation paths must be provided or test'
    train_split = (
        train_pickle_path is not None and validation_pickle_path is not None)

    if ((train_split and (not os.path.exists(train_pickle_path) or not os.path.exists(validation_pickle_path))) or
            (not train_split and not os.path.exists(test_pickle_path))):
        ocr_file_data, gs_file_data, ocr_tokens, gs_tokens = read_data(
            tokenize_service,
            data_service,
            full_data_path,
            full_ocr_path,
            full_gs_path,
            full_ocr_tokens_path,
            full_gs_tokens_path)

        token_pairs, decoded_pairs, _, _ = parse_metrics_obj(
            tokenize_service,
            metrics_service,
            ocr_tokens,
            gs_tokens,
            ocr_file_data,
            gs_file_data,
            pickles_path)

        if train_split:
            eval_indices = random.sample(
                range(len(token_pairs)), int(0.01 * len(token_pairs)))

            train_pairs = []
            eval_pairs = [[token_pairs[i], decoded_pairs[i]]
                          for i in eval_indices]

            eval_indices_dict = {i: False for i in range(len(token_pairs))}
            for i in eval_indices:
                eval_indices_dict[i] = True

            train_pairs = [[token_pairs[i], decoded_pairs[i]]
                           for i in range(len(token_pairs)) if not eval_indices_dict[i]]

            with open(train_pickle_path, 'wb') as train_handle:
                pickle.dump(train_pairs, train_handle, protocol=-1)

            with open(validation_pickle_path, 'wb') as eval_handle:
                pickle.dump(eval_pairs, eval_handle, protocol=-1)
        else:
            test_pairs = [[token_pair, decoded_pair]
                          for token_pair, decoded_pair
                          in zip(token_pairs, decoded_pairs)]

            with open(test_pickle_path, 'wb') as test_handle:
                pickle.dump(test_pairs, test_handle, protocol=-1)
    else:
        if train_split:
            with open(train_pickle_path, 'rb') as train_pickle_file:
                train_pairs = pickle.load(train_pickle_file)

            with open(validation_pickle_path, 'rb') as validation_pickle_file:
                eval_pairs = pickle.load(validation_pickle_file)
        else:
            with open(test_pickle_path, 'rb') as test_pickle_file:
                test_pairs = pickle.load(test_pickle_file)

    if train_split:
        return train_pairs, eval_pairs
    else:
        return test_pairs


def parse_language_data(
        tokenize_service: BaseTokenizeService,
        metrics_service: MetricsService,
        vocabulary_service: VocabularyService,
        data_service: DataService,
        pickles_path: str,
        full_data_path: str,
        split_data: bool = True) -> LanguageData:

    train_pickle_path = None
    validation_pickle_path = None
    test_pickle_path = None
    if split_data:
        train_pickle_path = os.path.join(pickles_path, 'train_pairs.pickle')
        validation_pickle_path = os.path.join(
            pickles_path, 'eval_pairs.pickle')
    else:
        test_pickle_path = os.path.join(pickles_path, 'test_pairs.pickle')

    full_ocr_path = os.path.join(pickles_path, 'combined_ocr.pickle')
    full_gs_path = os.path.join(pickles_path, 'combined_gs.pickle')

    full_ocr_tokens_path = os.path.join(
        pickles_path, 'combined_ocr_tokens.pickle')
    full_gs_tokens_path = os.path.join(
        pickles_path, 'combined_gs_tokens.pickle')

    data_split = load_split_data(
        tokenize_service,
        metrics_service,
        data_service,
        full_data_path,
        full_ocr_path,
        full_gs_path,
        full_ocr_tokens_path,
        full_gs_tokens_path,
        pickles_path,
        train_pickle_path,
        validation_pickle_path,
        test_pickle_path)

    if split_data:
        train_pairs, validation_pairs = data_split
        train_language_data = LanguageData([], [], [], [], [], [], [])
        for train_pair in train_pairs:
            train_language_data.add_entry(
                None, train_pair[0][0], train_pair[0][1], train_pair[1][0], train_pair[1][1], tokenize_service, vocabulary_service)

        validation_language_data = LanguageData([], [], [], [], [], [], [])
        for validation_pair in validation_pairs:
            validation_language_data.add_entry(
                None, validation_pair[0][0], validation_pair[0][1], validation_pair[1][0], validation_pair[1][1], tokenize_service, vocabulary_service)
        return train_language_data, validation_language_data
    else:
        test_language_data = LanguageData([], [], [], [], [], [], [])
        for test_pair in data_split:
            test_language_data.add_entry(
                None, test_pair[0][0], test_pair[0][1], test_pair[1][0], test_pair[1][1], tokenize_service, vocabulary_service)

        return test_language_data
