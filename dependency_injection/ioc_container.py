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
from models.transformer_fine_tune.kbert_model import KBertModel
from models.transformer_fine_tune.kxlnet_model import KXLNetModel
from models.multifit.multifit_model import MultiFitModel
from models.rnn_encoder_decoder.sequence_model import SequenceModel
from models.transformer_encoder_decoder.transformer_model import TransformerModel
from models.joint_model import JointModel
from models.ner_rnn.ner_rnn_model import NERRNNModel
from models.rnn_char_to_char.char_to_char_model import CharToCharModel

from optimizers.optimizer_base import OptimizerBase
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.adamw_optimizer import AdamWOptimizer
from optimizers.joint_adamw_optimizer import JointAdamWOptimizer

from services.arguments.postocr_arguments_service import PostOCRArgumentsService
from services.arguments.postocr_characters_arguments_service import PostOCRCharactersArgumentsService
from services.arguments.transformer_arguments_service import TransformerArgumentsService
from services.arguments.ner_arguments_service import NERArgumentsService
from services.arguments.semantic_arguments_service import SemanticArgumentsService
from services.arguments.arguments_service_base import ArgumentsServiceBase

from services.process.process_service_base import ProcessServiceBase
from services.process.ner_process_service import NERProcessService

from services.evaluation.base_evaluation_service import BaseEvaluationService
from services.evaluation.semantic_change_evaluation_service import SemanticChangeEvaluationService
from services.evaluation.ner_evaluation_service import NEREvaluationService

from services.data_service import DataService
from services.dataloader_service import DataLoaderService
from services.dataset_service import DatasetService
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
from services.plot_service import PlotService
from services.experiment_service import ExperimentService
from services.decoding_service import DecodingService

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
        elif configuration == Configuration.SequenceToCharacter or configuration == Configuration.CharacterToCharacterEncoderDecoder:
            argument_service_type = PostOCRArgumentsService
        elif configuration == Configuration.CharacterToCharacter:
            argument_service_type = PostOCRCharactersArgumentsService
    elif challenge == Challenge.NamedEntityLinking or challenge == Challenge.NamedEntityRecognition:
        argument_service_type = NERArgumentsService
    elif challenge == Challenge.SemanticChange:
        argument_service_type = SemanticArgumentsService
    else:
        raise Exception('Challenge not supported')

    return argument_service_type


def register_optimizer(
        joint_model: bool,
        evaluate: bool,
        run_experiments: bool,
        challenge: Challenge,
        configuration: Configuration,
        model: ModelBase,
        arguments_service: ArgumentsServiceBase):
    if evaluate or run_experiments:
        return None

    optimizer = None

    if not joint_model:
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            optimizer = providers.Singleton(
                AdamWOptimizer,
                arguments_service=arguments_service,
                model=model
            )
        elif (configuration == Configuration.MultiFit or
              configuration == Configuration.SequenceToCharacter or
              configuration == Configuration.TransformerSequence or
              configuration == Configuration.CharacterToCharacter or
              configuration == Configuration.CharacterToCharacterEncoderDecoder):
            optimizer = providers.Singleton(
                AdamOptimizer,
                arguments_service=arguments_service,
                model=model
            )
        elif configuration == Configuration.RNNSimple:
            optimizer = providers.Singleton(
                AdamOptimizer,
                arguments_service=arguments_service,
                model=model
            )
    else:
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            optimizer = providers.Singleton(
                JointAdamWOptimizer,
                arguments_service=arguments_service,
                model=model
            )

    return optimizer


def register_loss(
        joint_model: bool,
        configuration: Configuration,
        arguments_service: ArgumentsServiceBase):
    loss_function = None

    if not joint_model:
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            loss_function = providers.Singleton(KBertLoss)
        elif (configuration == Configuration.MultiFit or
              configuration == Configuration.SequenceToCharacter or
              configuration == Configuration.CharacterToCharacter or
              configuration == Configuration.CharacterToCharacterEncoderDecoder):
            loss_function = providers.Singleton(SequenceLoss)
        elif configuration == Configuration.TransformerSequence:
            loss_function = providers.Singleton(TransformerSequenceLoss)
        elif configuration == Configuration.RNNSimple:
            loss_function = providers.Singleton(NERLoss)
    elif configuration == Configuration.KBert or configuration == Configuration.XLNet:
        loss_function = providers.Singleton(JointLoss)

    return loss_function


def register_evaluation_service(
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        plot_service: PlotService,
        metrics_service: MetricsService,
        process_service: ProcessServiceBase,
        joint_model: bool,
        configuration: Configuration):
    evaluation_service = None

    if (configuration == Configuration.KBert or configuration == Configuration.XLNet):
        evaluation_service = providers.Factory(
            SemanticChangeEvaluationService,
            arguments_service=arguments_service,
            file_service=file_service,
            plot_service=plot_service,
            metrics_service=metrics_service
        )
    elif configuration == Configuration.RNNSimple:
        evaluation_service = providers.Factory(
            NEREvaluationService,
            arguments_service=arguments_service,
            file_service=file_service,
            plot_service=plot_service,
            metrics_service=metrics_service,
            process_service=process_service
        )
    elif (configuration == Configuration.MultiFit or
          configuration == Configuration.SequenceToCharacter or
          configuration == Configuration.TransformerSequence or
          configuration == Configuration.CharacterToCharacter or
          configuration == Configuration.CharacterToCharacterEncoderDecoder):
        evaluation_service = providers.Factory(BaseEvaluationService)

    return evaluation_service


def register_model(
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        plot_service: PlotService,
        metrics_service: MetricsService,
        data_service: DataService,
        tokenizer_service: TokenizerService,
        log_service: LogService,
        vocabulary_service: VocabularyService,
        model_service: ModelService,
        process_service: ProcessServiceBase,
        pretrained_representations_service: PretrainedRepresentationsService,
        decoding_service: DecodingService,
        joint_model: bool,
        configuration: Configuration):

    if not joint_model:
        if configuration == Configuration.KBert or configuration == Configuration.XLNet:
            model = providers.Singleton(
                KBertModel if configuration == Configuration.KBert else KXLNetModel,
                arguments_service=arguments_service,
                data_service=data_service
            )
        elif (configuration == Configuration.MultiFit or
              configuration == Configuration.SequenceToCharacter or
              configuration == Configuration.TransformerSequence or
              configuration == Configuration.CharacterToCharacter or
              configuration == Configuration.CharacterToCharacterEncoderDecoder):

            if configuration == Configuration.MultiFit:
                model = providers.Singleton(
                    MultiFitModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    tokenizer_service=tokenizer_service,
                    metrics_service=metrics_service,
                    log_service=log_service
                )
            elif configuration == Configuration.SequenceToCharacter or configuration == Configuration.CharacterToCharacterEncoderDecoder:
                model = providers.Singleton(
                    SequenceModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    metrics_service=metrics_service,
                    vocabulary_service=vocabulary_service,
                    pretrained_representations_service=pretrained_representations_service,
                    decoding_service=decoding_service)
            elif configuration == Configuration.TransformerSequence:
                model = providers.Singleton(
                    TransformerModel,
                    arguments_service=arguments_service,
                    data_service=data_service,
                    vocabulary_service=vocabulary_service,
                    metrics_service=metrics_service,
                    log_service=log_service,
                    tokenizer_service=tokenizer_service
                )
            elif configuration == Configuration.CharacterToCharacter:
                model = providers.Singleton(
                    CharToCharModel,
                    arguments_service=arguments_service,
                    vocabulary_service=vocabulary_service,
                    data_service=data_service,
                    metrics_service=metrics_service,
                    pretrained_representations_service=pretrained_representations_service)
        elif configuration == Configuration.RNNSimple:
            model = providers.Singleton(
                NERRNNModel,
                arguments_service=arguments_service,
                data_service=data_service,
                metrics_service=metrics_service,
                process_service=process_service,
                tokenizer_service=tokenizer_service
            )

    elif joint_model:
        model = providers.Singleton(
            JointModel,
            arguments_service=arguments_service,
            data_service=data_service,
            model_service=model_service
        )

    return model


def register_process_service(
        challenge: Challenge,
        arguments_service: ArgumentsServiceBase,
        file_service: FileService,
        tokenizer_service: TokenizerService):
    process_service = None
    if challenge == Challenge.NamedEntityLinking or challenge == Challenge.NamedEntityRecognition:
        process_service = providers.Singleton(
            NERProcessService,
            arguments_service=arguments_service,
            file_service=file_service,
            tokenizer_service=tokenizer_service)

    return process_service


class IocContainer(containers.DeclarativeContainer):
    """Application IoC container."""

    logger = providers.Singleton(logging.Logger, name='example')

    # Services

    arguments_service_base = ArgumentsServiceBase(
        raise_errors_on_invalid_args=False)

    challenge = arguments_service_base.challenge
    seed = arguments_service_base.seed
    device = arguments_service_base.device
    configuration = arguments_service_base.configuration
    joint_model = arguments_service_base.joint_model
    evaluate = arguments_service_base.evaluate
    run_experiments = arguments_service_base.run_experiments
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

    data_service = providers.Factory(DataService)

    file_service = providers.Factory(
        FileService,
        arguments_service=arguments_service
    )

    plot_service = providers.Factory(
        PlotService,
        data_service=data_service
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

    process_service = register_process_service(
        challenge,
        arguments_service,
        file_service,
        tokenizer_service)

    dataset_service = providers.Factory(
        DatasetService,
        arguments_service=arguments_service,
        mask_service=mask_service,
        tokenizer_service=tokenizer_service,
        file_service=file_service,
        log_service=log_service,
        pretrained_representations_service=pretrained_representations_service,
        vocabulary_service=vocabulary_service,
        metrics_service=metrics_service,
        data_service=data_service,
        process_service=process_service
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

    decoding_service = providers.Factory(
        DecodingService,
        arguments_service=arguments_service,
        vocabulary_service=vocabulary_service)

    model = register_model(
        arguments_service=arguments_service,
        file_service=file_service,
        plot_service=plot_service,
        metrics_service=metrics_service,
        data_service=data_service,
        tokenizer_service=tokenizer_service,
        log_service=log_service,
        vocabulary_service=vocabulary_service,
        model_service=model_service,
        process_service=process_service,
        pretrained_representations_service=pretrained_representations_service,
        decoding_service=decoding_service,
        joint_model=joint_model,
        configuration=configuration)

    loss_function = register_loss(
        joint_model=joint_model,
        configuration=configuration,
        arguments_service=arguments_service)

    optimizer = register_optimizer(
        joint_model,
        evaluate,
        run_experiments,
        challenge,
        configuration,
        model,
        arguments_service
    )

    evaluation_service = register_evaluation_service(
        arguments_service=arguments_service,
        file_service=file_service,
        plot_service=plot_service,
        metrics_service=metrics_service,
        process_service=process_service,
        joint_model=joint_model,
        configuration=configuration)

    experiment_service = providers.Factory(
        ExperimentService,
        arguments_service=arguments_service,
        metrics_service=metrics_service,
        file_service=file_service,
        tokenizer_service=tokenizer_service,
        vocabulary_service=vocabulary_service,
        plot_service=plot_service,
        data_service=data_service,
        model=model
    )

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
        test_service=test_service,
        experiment_service=experiment_service
    )
