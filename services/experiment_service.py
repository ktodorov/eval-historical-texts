import os
from typing import List, Tuple

import numpy as np
import torch
# import sklearn.manifold.t_sne
from MulticoreTSNE import MulticoreTSNE as TSNE

from enums.experiment_type import ExperimentType

from models.model_base import ModelBase

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.tokenizer_service import TokenizerService
from services.vocabulary_service import VocabularyService
from services.plot_service import PlotService
from services.data_service import DataService


class ExperimentService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            metrics_service: MetricsService,
            file_service: FileService,
            tokenizer_service: TokenizerService,
            vocabulary_service: VocabularyService,
            plot_service: PlotService,
            data_service: DataService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._file_service = file_service
        self._tokenizer_service = tokenizer_service
        self._vocabulary_service = vocabulary_service
        self._plot_service = plot_service
        self._data_service = data_service

        self._model = model.to(arguments_service.device)

    def execute_experiments(self, experiment_types: List[ExperimentType]):
        checkpoints_path = self._file_service.get_checkpoints_path()
        self._model.load(checkpoints_path, 'BEST')
        self._model.eval()

        if ExperimentType.WordSimilarity in experiment_types:
            self._calculate_word_similarity()

    def _calculate_word_similarity(self):
        words = self._vocabulary_service.get_all_english_nouns()
        if words is None:
            raise Exception('Words not found')

        language_pickle_path = os.path.join(
            self._file_service.get_pickles_path(),
            str(self._arguments_service.language))

        if not os.path.exists(language_pickle_path):
            os.mkdir(language_pickle_path)

        word_embeddings = self._get_word_embeddings(
            words, language_pickle_path)

        words_to_check = ['plane', 'awful', 'player', 'lass',
                          'edge', 'gas', 'contemplation', 'donkey', 'ball', 'attack']

        for word in words_to_check:
            self._calculate_word_closest(
                word, word_embeddings, words, language_pickle_path)

    def _get_word_embeddings(self, words: List[str], language_pickle_path: str):
        outputs_filename = f'word_embeddings_{self._arguments_service.checkpoint_name}'

        outputs = self._data_service.load_python_obj(
            language_pickle_path, outputs_filename)
        if outputs is None:
            outputs = np.zeros(
                (len(words), self._arguments_service.pretrained_model_size))

            for i, word in enumerate(words):
                if i % 100 == 0:
                    print(f'{i}/{len(words)}            \r', end='')

                outputs[i] = self._calculate_word_embeddings(word)

            self._data_service.save_python_obj(
                outputs,
                language_pickle_path,
                outputs_filename)

        return outputs

    def _calculate_word_closest(
            self,
            target_word: str,
            all_word_embeddings: np.array,
            all_words: List[str],
            language_pickle_path: str = None,
            word_amount: int = 5):
        target_word_embeddings = self._calculate_word_embeddings(target_word)
        closest_words, closest_word_embeddings = self._get_closest_words(
            target_word_embeddings, target_word, all_word_embeddings, all_words, word_amount)
        print(f'Closest words for \'{target_word}\': {closest_words}')
        self._plot_word_similarity(target_word_embeddings, target_word,
                                   closest_word_embeddings, closest_words, language_pickle_path)

    def _plot_word_similarity(
            self,
            target_word_embeddings: np.array,
            target_word: str,
            closest_word_embeddings,
            closest_words: List[str],
            save_path: str = None):
        all_word_embeddings = np.append(
            closest_word_embeddings, target_word_embeddings, axis=0)
        all_words = closest_words + [target_word]

        tsne = TSNE(n_components=2, random_state=0, n_jobs=4)
        tsne_result = tsne.fit_transform(all_word_embeddings)
        x_coords = tsne_result[:, 0]
        y_coords = tsne_result[:, 1]
        self._plot_service.plot_scatter(
            x_coords,
            y_coords,
            all_words,
            title=f'\'{target_word}\' similarity (Corpus #{self._arguments_service.checkpoint_name})',
            save_path=save_path,
            filename=f'{target_word}-similarity-{self._arguments_service.checkpoint_name}')

    def _calculate_word_embeddings(self, word: str):
        word_tokens, _, _, _ = self._tokenizer_service.encode_sequence(
            word)

        word_tokens_tensor = torch.tensor(word_tokens).unsqueeze(0).to(
            self._arguments_service.device)

        output = self._model.forward(word_tokens_tensor)
        return output.mean(dim=1).detach().cpu().numpy()

    def _get_closest_words(
            self,
            target_word_embeddings: np.array,
            target_word: str,
            all_word_embeddings: np.array,
            all_words: List[str],
            word_amount: int) -> Tuple[List[str], np.array]:
        cosine_distances = np.array([
            self._metrics_service.calculate_cosine_distance(target_word_embeddings, x) for x in all_word_embeddings
        ])

        indices = cosine_distances.argsort()[:word_amount][::-1]
        return [all_words[i] for i in indices], all_word_embeddings[indices]
