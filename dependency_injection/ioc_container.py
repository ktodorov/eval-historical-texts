import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main

from losses.loss_base import LossBase

from models.model_base import ModelBase

from services import config_service as config_service_namespace
from services import data_service as data_service_namespace
from services import dataset_service as dataset_service_namespace
from services import arguments_service as arguments_service_namespace
from services import dataloader_service as dataloader_service_namespace
from services import optimizer_service as optimizer_service_namespace
from services import train_service as train_service_namespace


import logging


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    config = providers.Configuration('config')
    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    config_service = providers.Singleton(
        config_service_namespace.ConfigService,
        config=config
    )

    data_service = providers.Factory(
        data_service_namespace.DataService,
        config_service=config_service,
        logger=logger,
    )

    arguments_service = providers.Singleton(
        arguments_service_namespace.ArgumentsService
    )

    dataset_service = providers.Factory(
        dataset_service_namespace.DatasetService
    )

    dataloader_service = providers.Factory(
        dataloader_service_namespace.DataLoaderService,
        arguments_service=arguments_service,
        dataset_service=dataset_service
    )

    loss_function = providers.Singleton(
        LossBase
    )

    model = providers.Singleton(
        ModelBase
    )

    optimizer_service = providers.Singleton(
        optimizer_service_namespace.OptimizerService,
        arguments_service=arguments_service,
        model=model
    )

    train_service = providers.Factory(
        train_service_namespace.TrainService,
        arguments_service=arguments_service,
        dataloader_service=dataloader_service,
        loss_function=loss_function,
        optimizer=optimizer_service().optimizer,
        model=model
    )

    # Misc

    main = providers.Callable(
        main.main,
        data_service=data_service,
        arguments_service=arguments_service,
        train_service=train_service
    )
