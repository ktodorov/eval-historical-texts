import os
from typing import List, Tuple

import numpy as np
import torch
# import sklearn.manifold.t_sne
from MulticoreTSNE import MulticoreTSNE as TSNE

from enums.experiment_type import ExperimentType

from entities.word_neighborhood import WordNeighborhood
from entities.batch_representation import BatchRepresentation

from models.model_base import ModelBase

from services.arguments.pretrained_arguments_service import PretrainedArgumentsService
from services.metrics_service import MetricsService
from services.file_service import FileService
from services.tokenize.base_tokenize_service import BaseTokenizeService
from services.vocabulary_service import VocabularyService
from services.plot_service import PlotService
from services.data_service import DataService
from services.cache_service import CacheService


class ExperimentService:
    def __init__(
            self,
            arguments_service: PretrainedArgumentsService,
            metrics_service: MetricsService,
            file_service: FileService,
            tokenize_service: BaseTokenizeService,
            vocabulary_service: VocabularyService,
            plot_service: PlotService,
            data_service: DataService,
            cache_service: CacheService,
            model: ModelBase):

        self._arguments_service = arguments_service
        self._metrics_service = metrics_service
        self._file_service = file_service
        self._tokenize_service = tokenize_service
        self._vocabulary_service = vocabulary_service
        self._plot_service = plot_service
        self._data_service = data_service
        self._cache_service = cache_service

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

        language_experiment_path = self._get_experiment_path(
            ExperimentType.WordSimilarity)

        # word_embeddings = self._cache_service.get_item_from_cache(
        #     item_key=f'all-word-embeddings',
        #     callback_function=lambda: self._generate_word_embeddings(words, language_experiment_path))
        word_embeddings = self._generate_word_embeddings(words, language_experiment_path)

        words_to_check = ['plane', 'player', 'gas',
                          'broadcast', 'awful', 'gay']  # , 'player', 'lass',
        # 'edge', 'gas', 'contemplation', 'donkey', 'ball', 'attack']

        for word in words_to_check:
            self._calculate_word_closest(
                word, word_embeddings, words, language_experiment_path)

    def _calculate_word_closest(
            self,
            target_word: str,
            all_word_embeddings: np.array,
            all_words: List[str],
            language_experiment_path: str = None,
            word_amount: int = 5):
        target_word_embeddings = self._calculate_word_embeddings(target_word)
        word_neighborhoods = []

        for i in range(len(target_word_embeddings)):
            word_neighborhood = self._get_word_neighborhood(
                target_word_embeddings[i], target_word, all_word_embeddings[i], all_words, word_amount)
            print(
                f'Closest words for \'{target_word}\': {word_neighborhood.closest_words} for Corpus #{i+1}')
            word_neighborhoods.append(word_neighborhood)

        self._plot_word_similarity(
            word_neighborhoods, language_experiment_path)

    def _generate_word_embeddings(self, words: List[str], language_experiment_path: str):
        # embeddings_size = self._arguments_service.pretrained_model_size
        embeddings_size = 300
        outputs_1 = np.zeros((len(words), embeddings_size))
        outputs_2 = np.zeros((len(words), embeddings_size))

        for i, word in enumerate(words):
            word_embeddings = self._calculate_word_embeddings(word)
            outputs_1[i], outputs_2[i] = word_embeddings

        return outputs_1, outputs_2

    def _plot_word_similarity(
            self,
            word_neighborhoods: List[WordNeighborhood],
            save_path: str = None):

        all_words = word_neighborhoods[0].closest_words
        all_words.append(word_neighborhoods[0].target_word_string)

        all_word_embeddings = word_neighborhoods[0].closest_word_embeddings
        all_word_embeddings = np.append(
            all_word_embeddings, word_neighborhoods[0].target_word_embeddings, axis=0)

        for word_neighborhood in word_neighborhoods[1:]:
            all_word_embeddings = np.concatenate(
                (all_word_embeddings, word_neighborhood.closest_word_embeddings), axis=0)
            all_word_embeddings = np.append(
                all_word_embeddings, word_neighborhood.target_word_embeddings, axis=0)

            all_words.extend(word_neighborhood.closest_words)
            all_words.append(word_neighborhood.target_word_string)

        tsne = TSNE(n_components=2, random_state=0, n_jobs=4)
        tsne_result = tsne.fit_transform(all_word_embeddings)

        ax = self._plot_service.create_plot()
        word_amount = []

        target_words_coords = []
        labels_colors = ['crimson', 'darkgreen']
        for i, word_neighborhood in enumerate(word_neighborhoods):
            x_coords = tsne_result[(i*6):((i+1)*6), 0]
            y_coords = tsne_result[(i*6):((i+1)*6), 1]
            target_words_coords.append([x_coords[-1], y_coords[-1]])
            bold_mask = [False for x in range(len(x_coords))]
            bold_mask[-1] = True
            self._plot_service.plot_scatter(
                x_coords,
                y_coords,
                color='white',
                ax=ax,
                show_plot=False)

            self._plot_service.plot_labels(
                x_coords,
                y_coords,
                all_words[(i*6):((i+1)*6)],
                color=labels_colors[i],
                ax=ax,
                show_plot=False,
                bold_mask=bold_mask)

        target_word = word_neighborhoods[0].target_word_string
        self._plot_service.plot_arrow(
            target_words_coords[0][0],
            target_words_coords[0][1],
            (target_words_coords[1][0] - target_words_coords[0][0]) * 0.95,
            (target_words_coords[1][1] - target_words_coords[0][1]) * 0.95,
            ax=ax,
            color='teal',
            title=f'Neighborhood change - \'{target_word.capitalize()}\'',
            save_path=save_path,
            filename=f'{target_word}-neighborhood-change',
            hide_axis=True)

    def _calculate_word_embeddings(self, word: str) -> List[np.array]:
        # word_tokens, _, _, _ = self._tokenize_service.encode_sequence(
        #     word)

        word_id = self._vocabulary_service.string_to_id(word)

        batch = BatchRepresentation(
            device=self._arguments_service.device,
            batch_size=1,
            word_sequences=[[word_id]])

        outputs = self._model.forward(batch)
        return [output.mean(dim=1).detach().cpu().numpy() for output in outputs]

    def _get_word_neighborhood(
            self,
            target_word_embeddings: np.array,
            target_word: str,
            all_word_embeddings: np.array,
            all_words: List[str],
            word_amount: int) -> WordNeighborhood:

        word_neighborhood = WordNeighborhood(
            target_word, target_word_embeddings, word_amount)
        word_neighborhood.calculate_word_neighbours(
            self._metrics_service, all_words, all_word_embeddings)
        return word_neighborhood

    def _get_experiment_path(self, experiment_type: ExperimentType):
        experiments_path = os.path.join(
            self._file_service.get_experiments_path(),
            str(experiment_type))

        if not os.path.exists(experiments_path):
            os.mkdir(experiments_path)

        language_experiment_path = os.path.join(
            experiments_path, str(self._arguments_service.language))

        if not os.path.exists(language_experiment_path):
            os.mkdir(language_experiment_path)

        return language_experiment_path
