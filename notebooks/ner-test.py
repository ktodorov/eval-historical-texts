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


def set_system_arguments(specific_args, language, seed, checkpoint_name, split_type):
    system_arguments = deepcopy(main_arguments)
    system_arguments.extend(specific_args)
    system_arguments.extend(language_args[language])
    system_arguments.extend(['--seed', str(seed)])

    system_arguments.extend(['--checkpoint-name', checkpoint_name])
    system_arguments.extend(['--resume-checkpoint-name', checkpoint_name])

    system_arguments.extend(['--split-type', split_type])

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
    'none-no-char': {
        'base_config': 'all--bert-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64"
        ]
    },
    'none': {
        'base_config': 'all--bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32"
        ]
    },
    'fast-text': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-nonew-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e64-l1-bi-d0.80.0001-nonew-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",
        ]
    },
    'both-finetune': {
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.8-tune0.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.8-tune0.0001-nonew-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e64-l1-bi-d0.8-tune0.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

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
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e64-l1-bi-d0.8-tune0.0001-nonew-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--hidden-dimension", "256",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-large-finetune': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e300-l1-bi-d0.8-tune0.0001-nonew-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",
            "--hidden-dimension", "256",
            "--embeddings-size", "300",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",
            "--include-pretrained-model",

            "--fine-tune-pretrained",
        ]
    },
    'bert-large': {
        'base_config': 'all--bert-pretr-ce16-ch32-h256-e300-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "256",
            "--embeddings-size", "300",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
        ]
    },
    'none-large': {
        'base_config': 'all--bert-ce16-ch32-h512-e1068-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "1068",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32"
        ]
    },
    'fast-text-large': {
        'base_config': 'all--ft-bert-ce16-ch32-h512-e768-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine", "literal-coarse", "metonymic-fine", "metonymic-coarse", "component", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "768",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-fasttext-model",
        ]
    },
    
    'both-single-lit-fine': {
        'base_config': '1--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-fine",

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
    
    'both-single-lit-coarse': {
        'base_config': '2--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "literal-coarse",# "metonymic-fine", "metonymic-coarse", "component", "nested",

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
    
    'both-single-met-fine': {
        'base_config': '3--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "metonymic-fine",

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
    
    'both-single-met-coarse': {
        'base_config': '4--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "metonymic-coarse",# "component", "nested",

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
    
    'both-single-component': {
        'base_config': '5--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "component",

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
    
    'both-single-nested': {
        'base_config': '6--ft-bert-pretr-ce16-ch32-h512-e64-l1-bi-d0.80.0001-spl',
        'args': [
            "--entity-tag-types", "nested",

            "--learn-new-embeddings",

            "--hidden-dimension", "512",
            "--embeddings-size", "64",

            "--learn-character-embeddings",
            "--character-embeddings-size", "16",
            "--character-hidden-size", "32",

            "--include-pretrained-model",
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


def get_test_metrics(json_data, col_name, scoring_type):
    return [
        str(round(json_data[col_name]['ALL'][scoring_type]['P_micro'], 3))[1:],
        str(round(json_data[col_name]['ALL'][scoring_type]['R_micro'], 3))[1:],
        str(round(json_data[col_name]['ALL'][scoring_type]['F1_micro'], 3))[1:]
    ]


def get_test_scores(coarse_path, fine_path, results_path, conf_name):

    scores = []
    with open(coarse_path, 'r') as coarse_file:
        json_data = json.load(coarse_file)

        for col_name in ['NE-COARSE-LIT', 'NE-COARSE-METO']:
            for scoring_type in ['partial', 'strict']:
                scores.extend(get_test_metrics(
                    json_data, col_name, scoring_type))

    if fine_path is not None:
        with open(fine_path, 'r') as fine_file:
            json_data = json.load(fine_file)

            for col_name in ['NE-FINE-LIT', 'NE-FINE-METO', 'NE-FINE-COMP', 'NE-NESTED']:
                for scoring_type in ['partial', 'strict']:
                    scores.extend(get_test_metrics(
                        json_data, col_name, scoring_type))

    results_str = ' & '.join(scores)

    if not os.path.exists(results_path):
        with open(results_path, 'w') as results_file:
            results_file.write('conf-name,results\n')

    with open(results_path, 'a') as results_file:
        results_file.write(f'{conf_name},{results_str}\n')


languages = [
    # 'french',
    # 'german',
    'english'
]
seeds = [13, 7, 25]
split_types = [
    'document',
    'multi-segment',
    'segment'
]
split_types_map = {
    'document': 'd',
    'multi-segment': 'ms',
    'segment': 's'
}

for language in languages:
    coarse_file_path = os.path.join('..', 'results', 'named-entity-recognition', 'rnn-simple', language, 'results_nerc_coarse_LANG_all.json')
    fine_file_path = os.path.join('..', 'results', 'named-entity-recognition', 'rnn-simple', language, 'results_nerc_fine_LANG_all.json')
    result_file_path = os.path.join('..', 'results', 'named-entity-recognition', 'rnn-simple', language, 'results.csv')


    for split_type in split_types:
        for config_name, config_values in specific_args.items():
            for seed in seeds:
                try:
                    checkpoint_name = f'BEST_{language}-{config_values["base_config"]}-{split_types_map[split_type]}-seed{seed}'
                    print('\n-------------------------------\n')
                    print(f'Starting {checkpoint_name}...')

                    output_path = os.path.join('..', 'results', 'named-entity-recognition', 'rnn-simple', language, f'output-{checkpoint_name}.tsv')
                    if not os.path.exists(output_path):
                        checkpoints_path = os.path.join('..', 'results', 'named-entity-recognition', 'rnn-simple', language, f'{checkpoint_name}.pickle')
                        if not os.path.exists(checkpoints_path):
                            print('Model checkpoint not found')
                            continue

                        # set_system_arguments(config_values['args'], language, seed, checkpoint_name, split_type)
                        # (dataloader_service, model, file_service, evaluation_service, arguments_service) = create_services()
                        # dataloader = load_data(model, dataloader_service, file_service, arguments_service)
                        # evaluation = perform_evaluation(dataloader, model, evaluation_service)
                        # output_path = evaluation_service.save_results(evaluation)

                    perform_task(language, output_path, task='nerc_coarse')

                    if language != 'english':
                        perform_task(language, output_path, task='nerc_fine')

                    get_test_scores(coarse_file_path, fine_file_path if language != 'english' else None,
                                    result_file_path, f'{config_name}-{split_type}-seed{seed}')

                except KeyboardInterrupt as ki:
                    raise ki
                except Exception as exception:
                    raise exception
                    print(
                        f'Error occurred for [{language}, {seed}, {config_name}]:\n{exception}\nContinuing...')

# print(get_test_scores(coarse_file_path))
# print(get_test_scores(fine_file_path))