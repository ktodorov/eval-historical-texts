import os
import urllib.request
import random
from shutil import copyfile
from multiprocessing import Pool, TimeoutError
import functools
import sys
import pickle

from typing import List

from transformers import PreTrainedTokenizer

from entities.language_data import LanguageData
from utils import path_utils

from services.data_service import DataService

ocr_prefix = '[OCR_toInput] '
ocr_aligned_prefix = '[OCR_aligned] '
gs_prefix = '[ GS_aligned] '


def combine_data(
        data_service: DataService,
        output_path: str,
        newseye_path: str,
        trove_path: str):
    newseye_2017_path = os.path.join(newseye_path, '2017')
    newseye_2019_path = os.path.join(newseye_path, '2019')

    process_newseye_files(newseye_2017_path, output_path,
                          'newseye-2017', data_service)
    process_newseye_files(newseye_2019_path, output_path,
                          'newseye-2019', data_service, subfolder_to_use='train')

    if not os.path.exists(trove_path):
        os.mkdir(trove_path)
        download_trove_files(trove_path)

    process_trove_files(trove_path, output_path, 'trove', data_service)
    # combine_full_data(output_path)


def cut_string(text: str, chunk_length: int):
    string_chunks = [
        text[i:i+chunk_length].replace('#', '').replace('@', '')
        for i in range(0, len(text), chunk_length)]

    return string_chunks


def process_newseye_files(
        data_path: str,
        output_path: str,
        unique_prefix: str,
        data_service: DataService,
        start_position: int = 14,
        max_string_length: int = 50,
        subfolder_to_use: str = 'full'):
    ocr_sequences = []
    gs_sequences = []

    for subdir_name in os.listdir(data_path):
        if subdir_name != subfolder_to_use:
            continue

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
                    data_file_text = data_file.read().split('\n')
                    ocr_strings = cut_string(
                        data_file_text[1][start_position:], max_string_length)
                    gs_strings = cut_string(
                        data_file_text[2][start_position:], max_string_length)

                    ocr_sequences.extend(ocr_strings)
                    gs_sequences.extend(gs_strings)

    max_length = 100
    empty_sequences_indices = [True if (ocr_sequences[i] == '' or gs_sequences[i] == '' or len(
        ocr_sequences[i]) > max_length or len(gs_sequences[i]) > max_length) else False for i in range(len(ocr_sequences))]
    ocr_sequences = [ocr_sequences[i] for i in range(
        len(ocr_sequences)) if not empty_sequences_indices[i]]
    gs_sequences = [gs_sequences[i] for i in range(
        len(gs_sequences)) if not empty_sequences_indices[i]]

    data_service.save_python_obj(
        (ocr_sequences, gs_sequences),
        output_path,
        unique_prefix)


def process_trove_files(
        data_path: str,
        output_full_path: str,
        unique_prefix: str,
        data_service: DataService):
    title_prefix = '*$*OVERPROOF*$*'
    separator = '||@@||'

    ocr_sequences = []
    gs_sequences = []

    for data_file_name in os.listdir(data_path):

        data_file_path = os.path.join(data_path, data_file_name)
        with open(data_file_path, 'r', encoding='utf-8') as data_file:
            file_content_lines = data_file.readlines()
            for file_line in file_content_lines:
                if file_line.startswith(title_prefix) or file_line == separator:
                    continue

                text_strings = file_line.split(separator)
                text_strings = [text_string.replace(
                    '#', '') for text_string in text_strings]
                text_strings = [text_string.replace(
                    '@', '') for text_string in text_strings]
                text_strings = [text_string.replace(
                    '\n', '') for text_string in text_strings]

                ocr_sequences.append(text_strings[0])
                gs_sequences.append(text_strings[1])

    empty_sequences_indices = [True if ocr_sequences[i] ==
                               '' or gs_sequences[i] == '' else False for i in range(len(ocr_sequences))]
    ocr_sequences = [ocr_sequences[i] for i in range(
        len(ocr_sequences)) if not empty_sequences_indices[i]]
    gs_sequences = [gs_sequences[i] for i in range(
        len(gs_sequences)) if not empty_sequences_indices[i]]

    data_service.save_python_obj(
        (ocr_sequences, gs_sequences), output_full_path, 'trove-data', print_success=True)


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
