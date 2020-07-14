import wandb
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import subprocess
import os

from copy import deepcopy

plt.rcParams["axes.grid"] = False

import sys
sys.path.append('..')

from enums.evaluation_type import EvaluationType
from utils.dict_utils import update_dictionaries

main_arguments = [
    "--device cuda",
    "--data-folder", "..\\data",
    "--checkpoint-folder", "..\\results",
    "--epochs", "100000",
    "--eval-freq", "30",
    "--configuration", "rnn-simple",
    "--learning-rate", "1e-2",
    "--metric-types", "f1-score", "precision", "recall",
    "--challenge", "named-entity-recognition",
    "--batch-size", "1",
    "--resume-training",

    "--no-attention",

    "--fine-tune-learning-rate", "1e-4",
    "--pretrained-model-size", "768",
    "--pretrained-max-length", "512",
    "--fasttext-model-size", "300",

    "--bidirectional-rnn",
    "--number-of-layers", "1",

    "--replace-all-numbers",

    "--merge-subwords",
    "--evaluate",

]


def set_system_arguments(specific_args, language, seed, checkpoint_name):
    system_arguments = deepcopy(main_arguments)
    system_arguments.extend(specific_args)
    system_arguments.extend(language_args[language])
    system_arguments.extend(['--seed', str(seed)])

    system_arguments.extend(['--checkpoint-name', checkpoint_name])
    system_arguments.extend(['--resume-checkpoint-name', checkpoint_name])

    sys.argv = system_arguments


language_args = {
    'french': [
        "--language", "french",
        "--fasttext-model", "fr-model-skipgram-300minc20-ws5-maxn-6.bin",
        "--pretrained-weights", "bert-base-multilingual-cased"
    ],
    'german': [
        "--language", "german",
        "--fasttext-model", "de-model-skipgram-300-minc20-ws5-maxn-6.bin",
        "--pretrained-weights", "bert-base-german-cased"
    ],
    'english': [
        "--language", "english",
        "--fasttext-model", "en-model-skipgram-300-minc5-ws5-maxn-6.bin",
        "--pretrained-weights", "bert-base-cased"
    ]
}

specific_args = {
    'both-no-new-doc': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'none-no-char-doc': {
        'base_config': 'all--bert-h512-e64-l1-bi-d0.80.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64"
        ]
    },
    'none-doc': {
        'base_config': 'all--bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32"
        ]
    },
    'fast-text-doc': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-fasttext-model",
        ]
    },
    'both-doc': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'bert-doc': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
        ]
    },
    'bert-no-new-doc': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",
        ]
    },
    'both-finetune-doc': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'both-finetune-no-new-doc': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-doc': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-no-new-doc': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-d',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "document",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'none-no-char': {
        'base_config': 'all--bert-h512-e64-l1-bi-d0.80.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64"
        ]
    },
    'none': {
        'base_config': 'all--bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32"
        ]
    },
    'fast-text': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-fasttext-model",
        ]
    },
    'both': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'both-no-new': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'bert': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
        ]
    },
    'bert-no-new': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",
        ]
    },
    'both-finetune': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'both-finetune-no-new': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-finetune': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-no-new-finetune': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-ms',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "multi-segment",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'none-no-char-segment': {
        'base_config': 'all--bert-h512-e64-l1-bi-d0.80.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64"
        ]
    },
    'none-segment': {
        'base_config': 'all--bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32"
        ]
    },
    'fast-text-segment': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-fasttext-model",
        ]
    },
    'both-segment': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'both-no-new-segment': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'bert-segment': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
        ]
    },
    'bert-no-new-segment': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.80.0001-nonew-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",
        ]
    },
    'both-finetune-segment': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'both-finetune-no-new-segment': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-finetune-segment': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-no-new-finetune-segment': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e256-l1-bi-d0.8-tune0.0001-nonew-spl-s',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",
            "--split-type", "segment",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
}

def create_services():
    # Configure container:

    from dependency_injection.ioc_container import IocContainer
    container = IocContainer()
    dataloader_service = container.dataloader_service()
    model = container.model()
    file_service = container.file_service()
    evaluation_service = container.evaluation_service()
    arguments_service = container.arguments_service()

    return dataloader_service, model, file_service, evaluation_service, arguments_service


def load_data(model, dataloader_service, file_service, arguments_service):
    dataloader = dataloader_service.get_test_dataloader()
    checkpoints_path = file_service.get_checkpoints_path()
    model.load(checkpoints_path, 'BEST', checkpoint_name=arguments_service.checkpoint_name)
    model.eval()
    model.to(arguments_service.device)

    return dataloader

def perform_evaluation(dataloader, model, evaluation_service):

    evaluation = {}
    dataloader_length = len(dataloader)

    for i, batch in enumerate(dataloader):
        print(f'{i}/{dataloader_length}         \r', end='')

        outputs = model.forward(batch)

        batch_evaluation = evaluation_service.evaluate_batch(
            outputs,
            batch,
            EvaluationType.NamedEntityRecognitionMatch,
            i)

        update_dictionaries(evaluation, batch_evaluation)

    return evaluation

languages_prefixes = {
    'french': 'fr',
    'english': 'en',
    'german': 'de'
}

def perform_task(language, output_path, task):
    p = subprocess.Popen([
        "python",
        'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\challenges\\clef\\scorer\\CLEF-HIPE-2020-scorer\\clef_evaluation.py',
        '--ref',
        f'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\code\\eval-historical-texts\\data\\named-entity-recognition\\rnn-simple\\{language}\\HIPE-data-v1.3-test-{languages_prefixes[language]}.tsv',
        '--pred', f'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\code\\eval-historical-texts\\notebooks\\{output_path}',
        '--task', task,
        '--skip_check'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, err = p.communicate()
    print(output.decode('utf-8'))
    error_str = err.decode('utf-8')
    print(error_str)
    assert p.returncode == 0


languages = ['french', 'german', 'english']
seeds = [13, 7, 25]

for language in languages:
    coarse_file_path = f'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\code\\eval-historical-texts\\results\\named-entity-recognition\\rnn-simple\\{language}\\results_nerc_coarse_LANG_all.json'
    fine_file_path = f'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\code\\eval-historical-texts\\results\\named-entity-recognition\\rnn-simple\\{language}\\results_nerc_fine_LANG_all.json'

    for config_name, config_values in specific_args.items():
        for seed in seeds:
            try:
                checkpoint_name = f'BEST_{language}-{config_values["base_config"]}-seed{seed}'
                print('\n-------------------------------\n')
                print(f'Starting {checkpoint_name}...')

                output_path = f'D:\\OneDrive\\Learning\\University\\Masters-UvA\\Thesis\\code\\eval-historical-texts\\results\\named-entity-recognition\\rnn-simple\\{language}\\output-{checkpoint_name}.tsv'
                if not os.path.exists(output_path):
                    set_system_arguments(config_values['args'], language, seed, checkpoint_name)
                    (dataloader_service, model, file_service, evaluation_service, arguments_service) = create_services()
                    dataloader = load_data(model, dataloader_service, file_service, arguments_service)
                    evaluation = perform_evaluation(dataloader, model, evaluation_service)
                    output_path = evaluation_service.save_results(evaluation)
                else:
                    print('Model is already tested')

            except KeyboardInterrupt as ki:
                raise ki
            except Exception as exception:
                print(f'Error occurred for [{language}, {seed}, {config_name}]:\n{exception}\nContinuing...')

# print(get_test_scores(coarse_file_path))
# print(get_test_scores(fine_file_path))


