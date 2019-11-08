import dependency_injector.containers as containers
import dependency_injector.providers as providers

import main
from services import config_service as config_service_namespace, data_service as data_service_namespace, arguments_service as arguments_service_namespace
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

    # Misc

    main = providers.Callable(
        main.main,
        data_service=data_service,
        arguments_service=arguments_service
    )
