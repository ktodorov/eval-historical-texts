import os
import sys
import pickle

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor

from enums.language import Language
from entities.language_data import LanguageData


def parse_language_data(
        dataset_folder_path: str,
        language: Language,
        tokenizer: SentencePieceProcessor) -> LanguageData:
    start_position = 14

    if not os.path.exists(dataset_folder_path):
        raise Exception('Folder path does not exist')

    for language_dir_name in os.listdir(dataset_folder_path):
        language_dir_path = os.path.join(
            dataset_folder_path, language_dir_name)

        if not os.path.isdir(language_dir_path):
            continue

        current_language = Language.from_str(language_dir_name)
        if current_language != Language.English:
            continue

        file_paths = []
        for subdir_name in os.listdir(language_dir_path):
            subdir_path = os.path.join(language_dir_path, subdir_name)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                file_paths.append(file_path)

        train_language_data = LanguageData()
        validation_language_data = LanguageData()

        split_index = int(len(file_paths) * 0.8)
        train_file_paths = file_paths[0:split_index]
        validation_file_paths = file_paths[split_index:]

        for file_path in train_file_paths:
            with open(file_path, 'r', encoding='utf-8') as language_file:
                text_data: List[str] = language_file.read().split('\n')

                train_language_data.add_entry(
                    text_data[0][start_position:],
                    text_data[1][start_position:],
                    text_data[2][start_position:],
                    tokenizer)

        for file_path in validation_file_paths:
            with open(file_path, 'r', encoding='utf-8') as language_file:
                text_data: List[str] = language_file.read().split('\n')

                validation_language_data.add_entry(
                    text_data[0][start_position:],
                    text_data[1][start_position:],
                    text_data[2][start_position:],
                    tokenizer)

        return train_language_data, validation_language_data

    return None, None


def train_spm_model(
        dataset_folder_path: str,
        data_output_path: str,
        language: Language,
        vocabulary_size: int):

    if not os.path.exists(dataset_folder_path):
        raise Exception('Folder path does not exist')

    for language_dir_name in os.listdir(dataset_folder_path):
        language_dir_path = os.path.join(
            dataset_folder_path, language_dir_name)

        if not os.path.isdir(language_dir_path):
            continue

        current_language = Language.from_str(language_dir_name)
        if current_language != Language.English:
            continue

        file_paths = ''
        for subdir_name in os.listdir(language_dir_path):
            subdir_path = os.path.join(language_dir_path, subdir_name)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                if file_paths:
                    file_paths = f'{file_paths},{file_path}'
                else:
                    file_paths = file_path

        model_prefix = os.path.join(data_output_path, 'tokenizer')
        spm.SentencePieceTrainer.Train(
            f'--input={file_paths} --model_prefix={model_prefix} --vocab_size={vocabulary_size}')


def preprocess_data(
        language: Language,
        train_data_path: str,
        test_data_path: str,
        data_output_path: str,
        tokenizer: SentencePieceProcessor):

    train_language_data, validation_language_data = parse_language_data(
        train_data_path, language, tokenizer)
    train_language_data_filepath = os.path.join(
        data_output_path, f'train_language_data.pickle')
    validation_language_data_filepath = os.path.join(
        data_output_path, f'validation_language_data.pickle')

    with open(train_language_data_filepath, 'wb') as handle:
        pickle.dump(train_language_data, handle, protocol=-1)

    with open(validation_language_data_filepath, 'wb') as handle:
        pickle.dump(validation_language_data, handle, protocol=-1)

    test_language_data = parse_language_data(
        test_data_path, language, tokenizer)
    test_language_data_filepath = os.path.join(
        data_output_path, f'test_language_data.pickle')

    with open(test_language_data_filepath, 'wb') as handle:
        pickle.dump(test_language_data, handle, protocol=-1)
