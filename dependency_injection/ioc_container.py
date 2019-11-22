import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main

from losses.loss_base import LossBase
from losses.kbert_loss import KBertLoss
from losses.joint_kbert_loss import JointKBertLoss

from models.model_base import ModelBase
from models.kbert_model import KBertModel
from models.joint_kbert_model import JointKBertModel

from optimizers.optimizer_base import OptimizerBase
from optimizers.adamw_optimizer import AdamWOptimizer
from optimizers.joint_adamw_optimizer import JointAdamWOptimizer

from services.arguments_service import ArgumentsService
from services.config_service import ConfigService
from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
from services.evaluation_service import EvaluationService
from services.log_service import LogService
from services.mask_service import MaskService
from services.test_service import TestService
from services.tokenizer_service import TokenizerService
from services.train_service import TrainService

import logging


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    config = providers.Configuration('config')
    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    log_service = providers.Singleton(
        LogService
    )

    config_service = providers.Singleton(
        ConfigService,
        config=config
    )

    data_service = providers.Factory(
        DataService,
        logger=logger,
    )

    arguments_service = providers.Singleton(
        ArgumentsService
    )

    tokenizer_service = providers.Singleton(
        TokenizerService,
        arguments_service=arguments_service,
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
        tokenizer_service=tokenizer_service
    )

    dataloader_service = providers.Factory(
        DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service
    )

    evaluation_service = providers.Factory(
        EvaluationService
    )

    configuration = arguments_service().get_argument('configuration')
    if configuration == 'kbert':
        loss_function = providers.Singleton(
            KBertLoss
        )

        model = providers.Singleton(
            KBertModel,
            arguments_service=arguments_service,
            data_service=data_service
        )

        optimizer = providers.Singleton(
            AdamWOptimizer,
            arguments_service=arguments_service,
            model=model
        )
    elif configuration == 'joint-kbert':
        loss_function = providers.Singleton(
            JointKBertLoss
        )

        model = providers.Singleton(
            JointKBertModel,
            arguments_service=arguments_service,
            data_service=data_service
        )

        optimizer = providers.Singleton(
            JointAdamWOptimizer,
            arguments_service=arguments_service,
            model=model
        )
    else:
        raise Exception('Unsupported configuration')

    test_service = providers.Factory(
        TestService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        evaluation_service=evaluation_service,
        model=model
    )

    train_service = providers.Factory(
        TrainService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        loss_function=loss_function,
        optimizer=optimizer,
        log_service=log_service,
        model=model
    )

    # Misc

    main = providers.Callable(
        main.main,
        data_service=data_service,
        arguments_service=arguments_service,
        train_service=train_service,
        test_service=test_service
    )
