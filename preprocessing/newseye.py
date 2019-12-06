import os
import sys
import pickle

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor

from enums.language import Language
from entities.language_data import LanguageData
from utils import path_utils


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

        language_data = LanguageData()

        for subdir_name in os.listdir(language_dir_path):
            subdir_path = os.path.join(language_dir_path, subdir_name)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as language_file:
                    text_data: List[str] = language_file.read().split('\n')

                    language_data.add_entry(
                        text_data[0][start_position:],
                        text_data[1][start_position:],
                        text_data[2][start_position:],
                        tokenizer)

        return language_data

    return None


def preprocess_data(
        language: Language,
        train_data_path: str,
        test_data_path: str,
        data_output_path: str,
        tokenizer: SentencePieceProcessor):
    train_language_data = parse_language_data(
        train_data_path, language, tokenizer)
    train_language_data_filepath = os.path.join(
        data_output_path, f'train_language_data.pickle')
    with open(train_language_data_filepath, 'wb') as handle:
        pickle.dump(train_language_data, handle, protocol=-1)

    test_language_data = parse_language_data(
        test_data_path, language, tokenizer)
    test_language_data_filepath = os.path.join(
        data_output_path, f'test_language_data.pickle')
    with open(test_language_data_filepath, 'wb') as handle:
        pickle.dump(test_language_data, handle, protocol=-1)


if __name__ == '__main__':
    data_folder_path = os.path.join(
        'data', 'ICDAR2019_POCR_competition_dataset')
    train_folder_path = os.path.join(
        data_folder_path, 'ICDAR2019_POCR_competition_training_18M_without_Finnish')

    eval_folder_path = os.path.join(
        data_folder_path, 'ICDAR2019_POCR_competition_evaluation_4M_without_Finnish')

    tokenizer = spm.SentencePieceProcessor()

    preprocess_data(Language.English, train_folder_path,
                    eval_folder_path, 'data', tokenizer)
