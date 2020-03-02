import torch
import numpy as np
import random

import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main

from enums.configuration import Configuration
from enums.challenge import Challenge

from losses.sequence_loss import SequenceLoss
from losses.transformer_sequence_loss import TransformerSequenceLoss
from losses.loss_base import LossBase
from losses.kbert_loss import KBertLoss
from losses.joint_loss import JointLoss
from losses.ner_loss import NERLoss

from models.model_base import ModelBase
from models.kbert_model import KBertModel
from models.kxlnet_model import KXLNetModel
from models.multifit_model import MultiFitModel
from models.sequence_model import SequenceModel
from models.transformer_model import TransformerModel
from models.joint_model import JointModel
from models.ner_rnn_model import NERRNNModel

from optimizers.optimizer_base import OptimizerBase
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.adamw_optimizer import AdamWOptimizer
from optimizers.joint_adamw_optimizer import JointAdamWOptimizer

from services.postocr_arguments_service import PostOCRArgumentsService
from services.transformer_arguments_service import TransformerArgumentsService
from services.ner_arguments_service import NERArgumentsService
from services.semantic_arguments_service import SemanticArgumentsService
from services.arguments_service_base import ArgumentsServiceBase
from services.config_service import ConfigService
from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.evaluation.semantic_change_evaluation_service import SemanticChangeEvaluationService
from services.file_service import FileService
from services.log_service import LogService
from services.mask_service import MaskService
from services.metrics_service import MetricsService
from services.model_service import ModelService
from services.pretrained_representations_service import PretrainedRepresentationsService
from services.test_service import TestService
from services.tokenizer_service import TokenizerService
from services.train_service import TrainService
from services.vocabulary_service import VocabularyService

import logging


def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def get_argument_service_type(challenge: Challenge, configuration: Configuration):
    argument_service_type = None
    if challenge == Challenge.PostOCRCorrection or challenge == Challenge.PostOCRErrorDetection:
        if configuration == Configuration.TransformerSequence:
            argument_service_type = TransformerArgumentsService
        else:
            argument_service_type = PostOCRArgumentsService
    elif challenge == Challenge.NamedEntityLinking or challenge == Challenge.NamedEntityRecognition:
        argument_service_type = NERArgumentsService
    elif challenge == Challenge.SemanticChange:
        argument_service_type = SemanticArgumentsService
    else:
        raise Exception('Challenge not supported')

    return argument_service_type


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    config = providers.Configuration('config')
    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    arguments_service_base = ArgumentsServiceBase(
        raise_errors_on_invalid_args=False)

    challenge = arguments_service_base.challenge
    seed = arguments_service_base.seed
    device = arguments_service_base.device
    configuration = arguments_service_base.configuration
    joint_model = arguments_service_base.joint_model
    external_logging_enabled = arguments_service_base.enable_external_logging

    argument_service_type = get_argument_service_type(challenge, configuration)
    arguments_service = providers.Singleton(
        argument_service_type
    )

    initialize_seed(seed, device)

    log_service = providers.Singleton(
        LogService,
        arguments_service=arguments_service,
        external_logging_enabled=external_logging_enabled
    )

    config_service = providers.Singleton(
        ConfigService,
        config=config
    )

    data_service = providers.Factory(DataService)

    file_service = providers.Factory(
        FileService,
        arguments_service=arguments_service
    )

    tokenizer_service = providers.Singleton(
        TokenizerService,
        arguments_service=arguments_service,
        file_service=file_service
    )

    mask_service = providers.Factory(
        MaskService,
        tokenizer_service=tokenizer_service,
        arguments_service=arguments_service
    )

    metrics_service = providers.Factory(
        MetricsService
    )

    vocabulary_service = providers.Singleton(
        VocabularyService,
        data_service=data_service,
        file_service=file_service
    )

    pretrained_representations_service = providers.Singleton(
        PretrainedRepresentationsService,
        arguments_service=arguments_service
    )

    dataset_service = providers.Factory(
        DatasetService,
        arguments_service=arguments_service,
        mask_service=mask_service,
        tokenizer_service=tokenizer_service,
        file_service=file_service,
        log_service=log_service,
        pretrained_representations_service=pretrained_representations_service,
        vocabulary_service=vocabulary_service,
        metrics_service=metrics_service
    )

    dataloader_service = providers.Factory(
        DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service
    )

    model_service = providers.Factory(
        ModelService,
        arguments_service=arguments_service,
        data_service=data_service
    )

    if not joint_model:
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            loss_function = providers.Singleton(
                KBertLoss
            )

            model = providers.Singleton(
                KBertModel if configuration == Configuration.KBert else KXLNetModel,
                arguments_service=arguments_service,
                data_service=data_service
            )

            optimizer = providers.Singleton(
                AdamWOptimizer,
                arguments_service=arguments_service,
                model=model
            )

            evaluation_service = providers.Factory(
                SemanticChangeEvaluationService,
                arguments_service=arguments_service,
                file_service=file_service
            )
        elif configuration == Configuration.MultiFit or configuration == Configuration.SequenceToCharacter or configuration == Configuration.TransformerSequence:

            if configuration == Configuration.MultiFit:
                loss_function = providers.Singleton(SequenceLoss)
                model = providers.Singleton(
                    MultiFitModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    tokenizer_service=tokenizer_service,
                    metrics_service=metrics_service,
                    log_service=log_service
                )
            elif configuration == Configuration.SequenceToCharacter:
                loss_function = providers.Singleton(SequenceLoss)
                model = providers.Singleton(
                    SequenceModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    tokenizer_service=tokenizer_service,
                    metrics_service=metrics_service,
                    log_service=log_service,
                    vocabulary_service=vocabulary_service
                )
            elif configuration == Configuration.TransformerSequence:
                loss_function = providers.Singleton(TransformerSequenceLoss)
                model = providers.Singleton(
                    TransformerModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    vocabulary_service=vocabulary_service,
                    metrics_service=metrics_service,
                    log_service=log_service,
                    tokenizer_service=tokenizer_service
                )

            optimizer = providers.Singleton(
                AdamOptimizer,
                arguments_service=arguments_service,
                model=model
            )

            evaluation_service = providers.Factory(
                BaseEvaluationService
            )
        elif configuration == Configuration.RNNSimple:
            loss_function = providers.Singleton(NERLoss)
            model = providers.Singleton(
                NERRNNModel,
                arguments_service=arguments_service,
                data_service=data_service,
                metrics_service=metrics_service
            )

            optimizer = providers.Singleton(
                AdamOptimizer,
                arguments_service=arguments_service,
                model=model
            )

            evaluation_service = providers.Factory(
                BaseEvaluationService
            )
    elif joint_model:

        model = providers.Singleton(
            JointModel,
            arguments_service=arguments_service,
            data_service=data_service,
            model_service=model_service
        )

        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            optimizer = providers.Singleton(
                JointAdamWOptimizer,
                arguments_service=arguments_service,
                model=model
            )

            loss_function = providers.Singleton(
                JointLoss
            )

            evaluation_service = providers.Factory(
                SemanticChangeEvaluationService,
                arguments_service=arguments_service,
                file_service=file_service
            )
        else:
            raise Exception(
                'No optimizer and loss defined for current configuration')
    else:
        raise Exception('Unsupported configuration')

    test_service = providers.Factory(
        TestService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        evaluation_service=evaluation_service,
        file_service=file_service,
        model=model
    )

    train_service = providers.Factory(
        TrainService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        loss_function=loss_function,
        optimizer=optimizer,
        log_service=log_service,
        model=model,
        file_service=file_service
    )

    # Misc

    main = providers.Callable(
        main.main,
        data_service=data_service,
        arguments_service=arguments_service,
        train_service=train_service,
        test_service=test_service
    )
