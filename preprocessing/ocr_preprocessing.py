import os
import urllib.request
import random
from shutil import copyfile
from multiprocessing import Pool, TimeoutError
import functools
import sys
import pickle

from transformers import PreTrainedTokenizer

from entities.language_data import LanguageData
from utils import path_utils

from services.vocabulary_service import VocabularyService

ocr_prefix = '[OCR_toInput] '
ocr_aligned_prefix = '[OCR_aligned] '
gs_prefix = '[ GS_aligned] '


def combine_data(ocr_path: str, newseye_path: str, trove_path: str):
    newseye_2017_path = os.path.join(newseye_path, '2017')
    newseye_2019_path = os.path.join(newseye_path, '2019')

    move_data(newseye_2017_path, ocr_path, 'newseye-2017')
    move_data(newseye_2019_path, ocr_path, 'newseye-2019')

    if not os.path.exists(trove_path):
        os.mkdir(trove_path)
        download_trove_files(trove_path)

    move_trove_data(trove_path, ocr_path, 'trove')
    combine_full_data(ocr_path)


def move_data(data_path: str, output_path: str, unique_prefix: str):
    for subdir_name in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for language_name in os.listdir(subdir_path):
            if not language_name.startswith('eng') and not language_name.startswith('EN'):
                continue

            language_path = os.path.join(subdir_path, language_name)
            for data_file_name in os.listdir(language_path):
                data_file_path = os.path.join(language_path, data_file_name)
                with open(data_file_path, 'r', encoding='utf-8') as data_file:
                    data_file_text = data_file.read()
                    data_file_text = data_file_text.replace('#', '')
                    data_file_text = data_file_text.replace('@', '')
                    output_file_path = os.path.join(
                        output_path, subdir_name, f'{unique_prefix}_{data_file_name}')
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(data_file_text)


def move_trove_data(data_path: str, output_path: str, unique_prefix: str):
    output_full_path = os.path.join(output_path, 'full')
    output_train_path = os.path.join(output_path, 'train')
    output_eval_path = os.path.join(output_path, 'eval')
    process_trove_files(data_path, output_full_path, unique_prefix)

    trove_file_names = list(filter(lambda x: x.startswith(
        'trove'), os.listdir(output_full_path)))

    eval_indices = random.sample(
        range(len(trove_file_names)), int(0.3 * len(trove_file_names)))

    for i, trove_file_name in enumerate(trove_file_names):
        trove_file_path = os.path.join(output_full_path, trove_file_name)
        if i in eval_indices:
            eval_file_path = os.path.join(output_eval_path, trove_file_name)
            copyfile(trove_file_path, eval_file_path)
        else:
            train_file_path = os.path.join(output_train_path, trove_file_name)
            copyfile(trove_file_path, train_file_path)


def download_trove_files(output_path: str):
    dataset1_file_urls = [
        f'http://overproof.projectcomputing.com/datasets/dataset1/rawTextAndHumanCorrectionPairs/smh{i}.txt' for i in range(1842, 1955)]

    for i, dataset1_file_url in enumerate(dataset1_file_urls):
        try:
            urllib.request.urlretrieve(
                dataset1_file_url, os.path.join(output_path, f'd1-{i}.txt'))
        except:
            print(
                f'There was error downloading file at url \'{dataset1_file_url}\'')
            continue

    dataset2_file_url = 'http://overproof.projectcomputing.com/datasets/dataset2/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt'
    urllib.request.urlretrieve(
        dataset2_file_url, os.path.join(output_path, f'd2.txt'))

    dataset3_file_url = 'http://overproof.projectcomputing.com/datasets/dataset3/rawTextAndHumanCorrectionAndOverproofCorrectionTriples/allArticles.txt'
    urllib.request.urlretrieve(
        dataset3_file_url, os.path.join(output_path, f'd3.txt'))


def process_trove_files(data_path: str, output_full_path: str, unique_prefix: str):
    title_prefix = '*$*OVERPROOF*$*'
    separator = '||@@||'

    counter = 0

    for data_file_name in os.listdir(data_path):
        articles_ocr = []
        articles_gs = []

        data_file_path = os.path.join(data_path, data_file_name)
        with open(data_file_path, 'r', encoding='utf-8') as data_file:
            file_content_lines = data_file.readlines()
            current_article_ocr = []
            current_article_gs = []
            for file_line in file_content_lines:

                if file_line.startswith(title_prefix):
                    if len(current_article_gs) > 0:
                        gs_text = ' '.join(current_article_gs)
                        articles_gs.append(gs_text)
                        ocr_text = ' '.join(current_article_ocr)
                        articles_ocr.append(ocr_text)

                        current_article_gs = []
                        current_article_ocr = []
                    continue

                if file_line == separator:
                    continue

                text_strings = file_line.split(separator)
                text_strings = [text_string.replace(
                    '#', '') for text_string in text_strings]
                text_strings = [text_string.replace(
                    '@', '') for text_string in text_strings]
                text_strings = [text_string.replace(
                    '\n', '') for text_string in text_strings]

                current_article_ocr.append(text_strings[0])
                current_article_gs.append(text_strings[1])

        for i in range(len(articles_gs)):
            with open(os.path.join(output_full_path, f'{unique_prefix}-{counter}.txt'), 'w', encoding='utf-8') as article_file:
                article_file.write(f'{ocr_prefix}\n')
                article_file.write(f'{ocr_aligned_prefix}{articles_ocr[i]}\n')
                article_file.write(f'{gs_prefix}{articles_gs[i]}\n')

            counter += 1


def combine_full_data(output_path: str):
    output_full_path = os.path.join(output_path, 'full')
    sub_files = os.listdir(output_full_path)
    full_text = ''

    pool = Pool(processes=4)
    copier = functools.partial(
        read_data_file, output_full_path=output_full_path)

    texts = pool.map(copier, sub_files)

    # for i, data_file_name in enumerate(sub_files):
    #     read_data_file(output_full_path, data_file_name)

    full_text = '\n'.join(texts)
    full_file_path = os.path.join(output_path, 'full.txt')
    with open(full_file_path, 'w', encoding='utf-8') as full_file:
        full_file.write(full_text)


def read_data_file(data_file_name: str, output_full_path: str):
    full_text = ''
    # print(f'\r{i}\\{len(sub_files)}         ', end='')
    data_file_path = os.path.join(output_full_path, data_file_name)
    with open(data_file_path, 'r', encoding='utf-8') as data_file:
        for line in data_file.readlines():
            line_text = line[len(ocr_prefix):]
            if line_text:
                full_text += f'{line_text}\n'

    return full_text


def parse_language_data(
        dataset_folder_path: str,
        tokenizer: PreTrainedTokenizer,
        vocabulary_service: VocabularyService) -> LanguageData:

    train_pickle_path = os.path.join(dataset_folder_path, 'train_pairs.pickle')
    with open(train_pickle_path, 'rb') as train_pickle_file:
        train_pairs = pickle.load(train_pickle_file)

    validation_pickle_path = os.path.join(dataset_folder_path, 'eval_pairs.pickle')
    with open(validation_pickle_path, 'rb') as validation_pickle_file:
        validation_pairs = pickle.load(validation_pickle_file)

#     start_position = 14

#     if not os.path.exists(dataset_folder_path):
#         raise Exception('Folder path does not exist')

#     file_paths = []
#     for file_name in os.listdir(dataset_folder_path):
#         file_path = os.path.join(dataset_folder_path, file_name)
#         file_paths.append(file_path)

    train_language_data = LanguageData([],[],[],[],[])
    for train_pair in train_pairs:
        train_language_data.add_entry(None, train_pair[0][0], train_pair[0][1], train_pair[1][0], train_pair[1][1], tokenizer, vocabulary_service)

    validation_language_data = LanguageData([],[],[],[],[])
    for validation_pair in validation_pairs:
        validation_language_data.add_entry(None, validation_pair[0][0], validation_pair[0][1], validation_pair[1][0], validation_pair[1][1], tokenizer, vocabulary_service)

#     split_index = int(len(file_paths) * 0.8)
#     train_file_paths = file_paths[0:split_index]
#     validation_file_paths = file_paths[split_index:]

#     pool = Pool(processes=4)

#     print('\rProcessing TRAIN data...', end='')
#     train_file_data = pool.map(read_ocr_file, train_file_paths)
#     for i, data in enumerate(train_file_data):
#         print(
#             f'\rProcessing TRAIN data...Saving ({i}\\{len(train_file_data)})', end='')
#         train_language_data.add_entry(data[0], data[1], data[2], tokenizer)
#     print('Processing TRAIN data...Done     ')

#     print('\rProcessing VALIDATION data...', end='')
#     validation_file_data = pool.map(read_ocr_file, validation_file_paths)
#     for i, data in enumerate(validation_file_data):
#         print(
#             f'\rProcessing VALIDATION data...Saving ({i}\\{len(validation_file_data)})', end='')
#         validation_language_data.add_entry(
#             data[0], data[1], data[2], tokenizer)
#     print('Processing VALIDATION data...Done     ')

    return train_language_data, validation_language_data


# start_position = 14


def read_ocr_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as language_file:
        text_data: List[str] = language_file.read().split('\n')

        return(
            text_data[0][start_position:],
            text_data[1][start_position:],
            text_data[2][start_position:])


def train_spm_model(
        dataset_folder_path: str,
        data_output_path: str,
        vocabulary_size: int):

    if not os.path.exists(dataset_folder_path):
        raise Exception('Folder path does not exist')

    model_prefix = os.path.join(data_output_path, 'tokenizer')
    spm.SentencePieceTrainer.Train(
        f'--input={dataset_folder_path} --model_prefix={model_prefix} --vocab_size={vocabulary_size}')


def preprocess_data(
        train_data_path: str,
        test_data_path: str,
        data_output_path: str,
        tokenizer: PreTrainedTokenizer,
        vocabulary_service: VocabularyService):

    train_language_data, validation_language_data = parse_language_data(
        train_data_path, tokenizer, vocabulary_service)
    train_language_data_filepath = os.path.join(
        data_output_path, f'train_language_data.pickle')
    validation_language_data_filepath = os.path.join(
        data_output_path, f'validation_language_data.pickle')

    with open(train_language_data_filepath, 'wb') as train_handle:
        pickle.dump(train_language_data, train_handle, protocol=-1)

    with open(validation_language_data_filepath, 'wb') as validation_handle:
        pickle.dump(validation_language_data, validation_handle, protocol=-1)

    # test_language_data = parse_language_data(
    #     test_data_path, tokenizer)
    # test_language_data_filepath = os.path.join(
    #     data_output_path, f'test_language_data.pickle')

    # with open(test_language_data_filepath, 'wb') as handle:
    #     pickle.dump(test_language_data, handle, protocol=-1)


if __name__ == '__main__':
    combine_data()
