import csv
import os

from typing import List, Dict
from overrides import overrides

import torch

from entities.batch_representation import BatchRepresentation

from enums.evaluation_type import EvaluationType
from enums.entity_tag_type import EntityTagType

from services.arguments.ner_arguments_service import NERArgumentsService
from services.file_service import FileService
from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.plot_service import PlotService
from services.metrics_service import MetricsService
from services.process.ner_process_service import NERProcessService


class NEREvaluationService(BaseEvaluationService):
    def __init__(
            self,
            arguments_service: NERArgumentsService,
            file_service: FileService,
            plot_service: PlotService,
            metrics_service: MetricsService,
            process_service: NERProcessService):
        super().__init__()

        self._arguments_service = arguments_service
        self._process_service = process_service
        self._file_service = file_service

    @overrides
    def evaluate_batch(
            self,
            output: torch.Tensor,
            batch_input: BatchRepresentation,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:
        predictions = output[0]

        result = []
        for entity_tag_type, type_predictions in predictions.items():
            for i, prediction in enumerate(type_predictions.squeeze(0).cpu().detach().tolist()):
                predicted_entity = self._process_service.get_entity_by_label(prediction, entity_tag_type, ignore_unknown=True)
                if len(result) <= i:
                    result.append({})

                result[i][entity_tag_type] = predicted_entity

        evaluation = {EvaluationType.NamedEntityRecognitionMatch: result}
        return evaluation

    @overrides
    def save_results(self, evaluation: Dict[EvaluationType, List]):
        data_path = self._file_service.get_data_path()
        language_suffix = self._process_service.get_language_suffix(
            self._arguments_service.language)
        dev_filepath = os.path.join(
            data_path, f'HIPE-data-v{self._process_service._data_version}-test-{language_suffix}.tsv')

        predictions = evaluation[EvaluationType.NamedEntityRecognitionMatch]

        tokens: List[str] = []
        with open(dev_filepath, 'r', encoding='utf-8') as dev_tsv:
            reader = csv.DictReader(dev_tsv, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
            header = reader.fieldnames
            for row in reader:
                tokens.append(row['TOKEN'])

        dev_word_amount = len([x for x in tokens if not x.startswith('#')])
        assert len(predictions) == dev_word_amount, f'Got "{len(predictions)}" predictions but expected "{dev_word_amount}"'

        column_mapping = {
            EntityTagType.Component: 'NE-FINE-COMP',
            EntityTagType.LiteralCoarse: 'NE-COARSE-LIT',
            EntityTagType.LiteralFine: 'NE-FINE-LIT',
            EntityTagType.MetonymicCoarse: 'NE-COARSE-METO',
            EntityTagType.MetonymicFine: 'NE-FINE-METO',
            EntityTagType.Nested: 'NE-NESTED',
        }

        checkpoints_path = self._file_service.get_checkpoints_path()
        file_path = os.path.join(
            checkpoints_path, f'output-{self._arguments_service.checkpoint_name}.tsv')
        with open(file_path, 'w', encoding='utf-8') as output_tsv:
            writer = csv.DictWriter(
                output_tsv, dialect=csv.excel_tab, fieldnames=header)
            writer.writeheader()

            counter = 0
            for token in tokens:
                if token.startswith('#'):
                    writer.writerow({'TOKEN': token})
                else:
                    row_dict = { column_mapping[entity_tag_type]: prediction for entity_tag_type, prediction in predictions[counter].items() }
                    row_dict['TOKEN'] = token
                    writer.writerow(row_dict)

                    counter += 1

        return file_path
