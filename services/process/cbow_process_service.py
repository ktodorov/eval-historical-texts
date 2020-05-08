import os
import pickle
import re
import numpy as np

from typing import List, Dict, Counter

from enums.language import Language

from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.file_service import FileService
from services.vocabulary_service import VocabularyService
from services.process.process_service_base import ProcessServiceBase


class CBOWProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            vocabulary_service: VocabularyService,
            file_service: FileService):
        super().__init__()

        self._vocabulary_service = vocabulary_service
        self._file_service = file_service

        self._pad_idx = 0
        self._unk_idx = 1
        self._cls_idx = 2
        self._eos_idx = 3
        self._mask_idx = 4

        self._window_size = 4

        corpus_id = arguments_service.corpus

        full_data_path = file_service.get_data_path()

        vocabulary = self._load_vocabulary(
            file_service,
            full_data_path,
            arguments_service.language)

        vocabulary_service.initialize_vocabulary_data(vocabulary)

        self._language = arguments_service.language
        self._corpus_id = corpus_id
        self._full_data_path = full_data_path

    def _load_vocabulary(
            self,
            file_service: FileService,
            full_data_path: str,
            language: Language):
        vocab_path = os.path.join(full_data_path, f'vocab.pickle')
        if not os.path.exists(vocab_path):
            challenge_path = file_service.get_challenge_path()
            semeval_data_path = os.path.join(challenge_path, 'eval')

            vocab = self._create_vocabulary(semeval_data_path, language)
            with open(vocab_path, 'wb') as vocab_file:
                pickle.dump(vocab, vocab_file)
        else:
            with open(vocab_path, 'rb') as data_file:
                vocab = pickle.load(data_file)

            return vocab

    def load_corpus_data(self, limit_size: int = None) -> list:
        corpus_data_path = os.path.join(
            self._full_data_path, f'corpus-{self._corpus_id}-data.pickle')
        if not os.path.exists(corpus_data_path):
            challenge_path = self._file_service.get_challenge_path()
            semeval_data_path = os.path.join(challenge_path, 'eval')

            corpus_data = self._preprocess_corpus_data(semeval_data_path)
            with open(corpus_data_path, 'wb') as corpus_data_file:
                pickle.dump(corpus_data, corpus_data_file)
        else:
            with open(corpus_data_path, 'rb') as corpus_data_file:
                corpus_data = pickle.load(corpus_data_file)

        if limit_size is not None:
            corpus_data = corpus_data[:limit_size]

        return corpus_data

    def _create_vocabulary(
            self,
            semeval_data_path: str,
            language: Language) -> Dict[str, int]:

        language_folder = os.path.join(semeval_data_path, language.value)
        corpus_ids = [1, 2]

        words_counter = Counter()
        threshold = 10

        targets_path = os.path.join(language_folder, 'targets.txt')
        with open(targets_path, 'r', encoding='utf-8') as targets_file:
            target_words = targets_file.readlines()

            # remove the POS tags if we are working with English
            if language == Language.English:
                target_words = [x[:-4] for x in target_words]

        for corpus_id in corpus_ids:
            corpus_path = os.path.join(
                language_folder,
                f'corpus{corpus_id}',
                'lemma')

            text_filenames = list(filter(
                lambda x: x.endswith('.txt'),
                os.listdir(corpus_path)))

            for text_filename in text_filenames:
                file_path = os.path.join(corpus_path, text_filename)
                with open(file_path, 'r', encoding='utf-8') as textfile:
                    file_text = textfile.read()
                    preprocessed_text = self._preprocess_text(file_text)
                    current_words = preprocessed_text.split(' ')
                    current_words_counter = Counter(current_words)
                    words_counter.update(current_words_counter)

        words = list(
            [word for word, count in words_counter.items() if count > threshold])

        # if any of the target words is now missing in the vocabulary, we make sure to add it back
        for target_word in target_words:
            if target_word not in words:
                words.append(target_word)

        vocabulary = {word: i+5 for i, word in enumerate(words)}

        vocabulary['[PAD]'] = self._pad_idx
        vocabulary['[UNK]'] = self._unk_idx
        vocabulary['[CLS]'] = self._cls_idx
        vocabulary['[EOS]'] = self._eos_idx
        vocabulary['[MASK]'] = self._mask_idx

        result = {
            'char2int': vocabulary,
            'int2char': {id: char for char, id in vocabulary.items()}
        }

        return result

    def _preprocess_text(self, text: str) -> str:
        result = text.lower().replace('\n', ' ')

        result = re.sub('(([0-9]+)|(([0-9]*)\.([0-9]*)))', '$NMB$', result)

        return result

    def _preprocess_corpus_data(self, semeval_data_path: str):
        corpus_path = os.path.join(
            semeval_data_path,
            self._language.value,
            f'corpus{self._corpus_id}',
            'lemma')

        text_filenames = list(filter(
            lambda x: x.endswith('.txt'),
            os.listdir(corpus_path)))

        cbow_entries: List[List[int]] = []

        for text_filename in text_filenames:
            file_path = os.path.join(corpus_path, text_filename)
            with open(file_path, 'r', encoding='utf-8') as textfile:
                file_text_lines = textfile.readlines()
                file_text_lines = [self._preprocess_file_text_line(
                    file_text_line) for file_text_line in file_text_lines]

                text_lines = len(file_text_lines)

                cbow_entries = [
                    self._vocabulary_service.string_to_ids(file_text_line)
                    for file_text_line in file_text_lines
                ]

        return cbow_entries

    def _preprocess_file_text_line(
            self,
            file_text_line: str):
        return self._preprocess_text(file_text_line).split(' ')

    def _get_cbow_entry(
            self,
            file_text_line):

        cbow_entries = []

        return cbow_entries
