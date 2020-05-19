import os
import pickle
import re
import numpy as np
import gensim

import torch
from typing import List, Dict, Counter

from enums.language import Language

from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.file_service import FileService
from services.vocabulary_service import VocabularyService
from services.data_service import DataService
from services.process.process_service_base import ProcessServiceBase

from nltk.tokenize import RegexpTokenizer

class CBOWProcessService(ProcessServiceBase):
    def __init__(
            self,
            arguments_service: SemanticArgumentsService,
            vocabulary_service: VocabularyService,
            file_service: FileService,
            data_service: DataService):
        super().__init__()

        self._vocabulary_service = vocabulary_service
        self._file_service = file_service
        self._data_service = data_service

        self._device = arguments_service.device

        self._pad_idx = 0
        self._unk_idx = 1
        self._cls_idx = 2
        self._eos_idx = 3
        self._mask_idx = 4

        self._window_size = 4

        self._tokenizer = RegexpTokenizer(r'\w+')

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
            corpus_data = (corpus_data[0][:limit_size],
                           corpus_data[1][:limit_size])

        return corpus_data

    def get_pretrained_embedding_weights(self) -> torch.Tensor:
        data_path = self._file_service.get_data_path()
        pretrained_weights_filename = 'pretrained-weights'
        pretrained_weights = self._data_service.load_python_obj(data_path, pretrained_weights_filename, print_on_error=False)
        if pretrained_weights is not None:
            pretrained_weights = pretrained_weights.float().to(self._device)
            return pretrained_weights

        word2vec_model_path = os.path.join(self._full_data_path, 'GoogleNews-vectors-negative300.bin')
        word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary = True)
        vocabulary_iterator = self._vocabulary_service.get_vocabulary_tokens()
        pretrained_weight_matrix = np.random.rand(
            self._vocabulary_service.vocabulary_size(),
            word2vec_weights.vector_size)

        for index, word in vocabulary_iterator:
            if word in word2vec_weights.index2word:
                pretrained_weight_matrix[index] = word2vec_weights.wv[word]

        result = torch.from_numpy(pretrained_weight_matrix).float().to(self._device)

        self._data_service.save_python_obj(result, data_path, pretrained_weights_filename, print_success=False)

        return result

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
                    current_words = self._preprocess_text(file_text)
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
        result = self._tokenizer.tokenize(text.lower().replace('\n', ' '))
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

                cbow_entries = []
                cbow_targets = []
                file_text_lines_len = len(file_text_lines)
                for i, file_text_line in enumerate(file_text_lines):
                    print(
                        f'Processing text lines: {i}/{file_text_lines_len}            \r', end='')
                    line_chunks, targets = self._split_line_to_chunks(
                        file_text_line)
                    cbow_entries.extend(line_chunks)
                    cbow_targets.extend(targets)

        return cbow_entries, cbow_targets

    def _preprocess_file_text_line(
            self,
            file_text_line: str):
        return list(filter(None, self._preprocess_text(file_text_line)))

    def _split_line_to_chunks(self, text_line: str):
        splitted_text_line = self._vocabulary_service.string_to_ids(text_line)
        window_size = 3

        chunks = []
        targets = []

        for i in range(window_size, len(splitted_text_line) - window_size - 1):
            context_words = splitted_text_line[i-window_size:i] + \
                splitted_text_line[i+1:i+window_size+1]

            chunks.append(context_words)
            targets.append(splitted_text_line[i])

        return chunks, targets
