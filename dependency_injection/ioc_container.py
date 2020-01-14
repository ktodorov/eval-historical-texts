import torch
import numpy as np
import random

import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main

from enums.configuration import Configuration

from losses.cross_entropy_loss import CrossEntropyLoss
from losses.loss_base import LossBase
from losses.kbert_loss import KBertLoss
from losses.joint_loss import JointLoss

from models.model_base import ModelBase
from models.kbert_model import KBertModel
from models.kxlnet_model import KXLNetModel
from models.multifit_model import MultiFitModel
from models.joint_model import JointModel

from optimizers.optimizer_base import OptimizerBase
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.adamw_optimizer import AdamWOptimizer
from optimizers.joint_adamw_optimizer import JointAdamWOptimizer

from services.arguments_service import ArgumentsService
from services.config_service import ConfigService
from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
from services.evaluation_service import EvaluationService
from services.file_service import FileService
from services.log_service import LogService
from services.mask_service import MaskService
from services.model_service import ModelService
from services.test_service import TestService
from services.tokenizer_service import TokenizerService
from services.train_service import TrainService

import logging


def initialize_seed(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    config = providers.Configuration('config')
    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    arguments_service = providers.Singleton(
        ArgumentsService
    )

    arguments_service_instance = arguments_service()
    initialize_seed(
        arguments_service_instance.get_argument('seed'),
        arguments_service_instance.get_argument('device'))

    external_logging_enabled: bool = arguments_service_instance.get_argument('enable_external_logging')

    log_service = providers.Singleton(
        LogService,
        arguments_service=arguments_service,
        external_logging_enabled=external_logging_enabled
    )

    config_service = providers.Singleton(
        ConfigService,
        config=config
    )

    data_service = providers.Factory(
        DataService,
        logger=logger,
    )

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

    dataset_service = providers.Factory(
        DatasetService,
        arguments_service=arguments_service,
        mask_service=mask_service,
        tokenizer_service=tokenizer_service,
        file_service=file_service,
        log_service=log_service
    )

    dataloader_service = providers.Factory(
        DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service
    )

    evaluation_service = providers.Factory(
        EvaluationService
    )

    model_service = providers.Factory(
        ModelService,
        arguments_service=arguments_service,
        data_service=data_service
    )

    configuration: Configuration = arguments_service_instance.get_argument(
        'configuration')
    joint_model: bool = arguments_service_instance.get_argument('joint_model')
    device: torch.device = arguments_service_instance.get_argument('device')
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
        elif configuration == Configuration.MultiFit:
            loss_function = providers.Singleton(
                CrossEntropyLoss,
                device=device
            )

            model = providers.Singleton(
                MultiFitModel,
                arguments_service=arguments_service,
                data_service=data_service,
                tokenizer_service=tokenizer_service
            )

            optimizer = providers.Singleton(
                AdamOptimizer,
                arguments_service=arguments_service,
                model=model
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
