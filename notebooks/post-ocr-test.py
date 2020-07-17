import sys
import wandb
import matplotlib.pyplot as plt
import scipy
import numpy as np
import json
import subprocess
import os

from copy import deepcopy

plt.rcParams["axes.grid"] = False

sys.path.append('..')
from utils.dict_utils import update_dictionaries
from enums.evaluation_type import EvaluationType

main_arguments = [
    "--device cuda",
    "--data-folder", "..\\data",
    "--checkpoint-folder", "..\\results",

    "--seed", "13",
    "--configuration", "char-to-char-encoder-decoder",
    "--challenge", "post-ocr-correction",
    "--batch-size", "32",

    "--evaluate",
    "--evaluation-type", "jaccard-similarity", "levenshtein-edit-distance-improvement",

    "--fine-tune-learning-rate", "1e-4",
    "--pretrained-model-size", "768",
    "--pretrained-max-length", "512",

    "--share-embedding-layer",
    "--learn-new-embeddings",
    "--number-of-layers", "2",
    "--dropout", "0.5",
    "--bidirectional",

    "--hidden-dimension", "512",
    "--encoder-embedding-size", "64",
    "--decoder-embedding-size", "64",
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
    'none': {
        'base_config': 'h512-e64-l2-bi-d0.50.0001',
        'args': []
    },
    'fast-text': {
        'base_config': 'ft-h512-e64-l2-bi-d0.50.0001',
        'args': [
            "--include-fasttext-model",
        ]
    },
    'both': {
        'base_config': 'pretr-ft-h512-e64-l2-bi-d0.50.0001',
        'args': [
            "--include-pretrained-model",
            "--include-fasttext-model",
        ]
    },
    'bert': {
        'base_config': 'pretr-h512-e64-l2-bi-d0.50.0001',
        'args': [
            "--include-pretrained-model",
        ]
    },
    'both-finetune': {
        'base_config': 'pretr-ft-h512-e64-l2-bi-d0.5-tune0.0001',
        'args': [
            "--include-pretrained-model",
            "--include-fasttext-model",

            "--fine-tune-pretrained",
            # // "--fine-tune-after-convergence",
        ]
    },
    'bert-finetune': {
        'base_config': 'pretr-h512-e64-l2-bi-d0.5-tune0.0001',
        'args': [
            "--include-pretrained-model",
            "--fine-tune-pretrained",
        ]
    },
    'bert-large': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e300-l1-bi-d0.80.0001-spl',
        'args': [
            "--encoder-embedding-size", "300",
            "--decoder-embedding-size", "300",
            "--include-pretrained-model",
        ]
    },
    'none-large': {
        'base_config': 'all--bert-ce16-ch32-h512-e1068-l1-bi-d0.80.0001-spl',
        'args': [
            "--encoder-embedding-size", "1068",
            "--decoder-embedding-size", "1068",
        ]
    },
    'fast-text-large': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e768-l1-bi-d0.80.0001-spl',
        'args': [
            "--encoder-embedding-size", "768",
            "--decoder-embedding-size", "768",
            "--include-fasttext-model",
        ]
    }
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
    model.load(checkpoints_path, 'BEST',
               checkpoint_name=arguments_service.checkpoint_name)
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
            [EvaluationType.JaccardSimilarity, EvaluationType.LevenshteinEditDistanceImprovement],
            i)

        update_dictionaries(evaluation, batch_evaluation)

    return evaluation


languages_prefixes = {
    'french': 'fr',
    'english': 'en',
    'german': 'de'
}

languages = [
    'french',
    'german',
    'english'
]
seeds = [13]#, 7, 25]

for language in languages:
    for config_name, config_values in specific_args.items():
        for seed in seeds:
            try:
                checkpoint_name = f'BEST_{language}--{config_values["base_config"]}-seed{seed}'
                print('\n-------------------------------\n')
                print(f'Starting {checkpoint_name}...')

                output_path = os.path.join('..', 'results', 'post-ocr-correction', 'char-to-char-encoder-decoder', language, 'output', f'output-{checkpoint_name}.csv')
                if not os.path.exists(output_path):
                    checkpoints_path = os.path.join('..', 'results', 'post-ocr-correction', 'char-to-char-encoder-decoder', language, f'{checkpoint_name}.pickle')
                    if not os.path.exists(checkpoints_path):
                        print('Model checkpoint not found')
                        continue

                    set_system_arguments(config_values['args'], language, seed, checkpoint_name)
                    (dataloader_service, model, file_service, evaluation_service, arguments_service) = create_services()
                    dataloader = load_data(model, dataloader_service, file_service, arguments_service)
                    evaluation = perform_evaluation(dataloader, model, evaluation_service)
                    evaluation_service.save_results(evaluation)

            except KeyboardInterrupt as ki:
                raise ki
            except Exception as exception:
                raise exception
                # print(f'Error occurred for [{language}, {seed}, {config_name}]:\n{exception}\nContinuing...')