import csv
import os

from typing import List, Dict

import torch

from enums.evaluation_type import EvaluationType
from enums.ner_type import NERType

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

    def evaluate_batch(
            self,
            output: torch.Tensor,
            evaluation_types: List[EvaluationType],
            batch_index: int) -> Dict[EvaluationType, List]:
        predictions = output[0].max(dim=1)[1].detach().tolist()
        position_changes = self._process_service.get_position_changes(
            batch_index)

        result = []
        for prediction in predictions:
            predicted_entity = self._process_service.get_entity_by_label(
                prediction)
            result.append(predicted_entity)

        if position_changes is not None:
            new_predictions = []
            for original_position, changes in position_changes.items():
                merged_prediction = None
                for change_position in changes:
                    if result[change_position] is not None:
                        merged_prediction = result[change_position]
                        break

                new_predictions.append(merged_prediction)

            result = new_predictions

        evaluation = {EvaluationType.NamedEntityRecognitionMatch: result}
        return evaluation

    def save_results(self, evaluation: Dict[EvaluationType, List], targets: List[str]):
        data_path = self._file_service.get_data_path()
        language_suffix = self._process_service.get_language_suffix(
            self._arguments_service.language)
        dev_filepath = os.path.join(
            data_path, f'HIPE-data-v1.1-dev-{language_suffix}.tsv')

        predictions = evaluation[EvaluationType.NamedEntityRecognitionMatch]

        tokens: List[str] = []
        with open(dev_filepath, 'r', encoding='utf-8') as dev_tsv:
            reader = csv.DictReader(dev_tsv, dialect=csv.excel_tab)
            header = reader.fieldnames
            for row in reader:
                tokens.append(row['TOKEN'])

        dev_word_amount = len([x for x in tokens if not x.startswith('#')])
        assert len(predictions) == dev_word_amount

        output_column = 'NE-COARSE-LIT' if self._arguments_service.label_type == NERType.Coarse else 'NE-FINE-LIT'

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
                    writer.writerow(
                        {'TOKEN': token, output_column: predictions[counter]})

                    counter += 1
