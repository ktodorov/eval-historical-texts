[< go back to main page](../../README.md)

# Base arguments

## Description

These arguments are generic and applicable to every configuration and challenge.

## Argument list

| Parameter     | Type          | Default value  | Description |
| ------------- | ------------- | -------------- |-------------|
| `epochs` | `int` | 500 | Maximum number of epochs |
| `eval-freq` | `int` | 50 | Evaluate every x batches |
| `batch-size` | `int` | 8 | Size of batches |
| `max-training-minutes` | `int` | 1440 (6 days) | Maximum minutes of training before save-and-kill |
| `device` | `str` | cuda | Device to be used. Pick from `cpu`/`cuda`. If default none is used automatic check will be done |
| `seed` | `int` | 42 | Random seed |
| `evaluate` | `bool` | False | Run in evaluation mode |
| `patience` | `int` | 30 | How long will the model wait for improvement before stopping training |
| `language` | [`Language`](../../enums/language.py) | English | Which language to train on |
| `shuffle` | `bool` | True | Shuffle training dataset |
| `learning-rate` | `float` | 1e-5 | Learning rate for training models |
| `weight-decay` | `float` | 1e-8 | Weight decay for optimizer |
| `momentum` | `float` | 0 | Momentum for optimizer |
| `checkpoint-name` | `str` | None | name that can be used to distinguish checkpoints |
| `resume-training` | `bool` | False | Resume training using saved checkpoints |
| `resume-checkpoint-name` | `str` | None | Checkpoint name that will be used to resume training from. If None is given, then current checkpoint name will be used |
| `skip-best-metrics-on-resume` | `bool` | False | Whether to skip loading saved metrics and continuing from last best checkpoint |
| `data-folder` | `str` | data | Folder where data will be taken from |
| `output-folder` | `str` | results | Folder where results and checkpoints will be saved |
| `checkpoint-folder` | `str` | None | Folder where checkpoints will be saved/loaded. If it is not provided, the output folder will be used |
| `evaluation-type` | [`EvaluationType`](../../enums/evaluation_type.py) | None | What type of evaluations should be performed |
| `output-eval-format` | [`OutputFormat`](../../enums/output_format.py) | None | What the format of the output after evaluation will be |
| `challenge` | [`Challenge`](../../enums/challenge.py) | None | Challenge that the model is being trained for. Data and output results will be put into such specific folder |
| `configuration` | [`Configuration`](../../enums/configuration.py) | KBert | Which configuration of model to load and use |
| `metric-types` | [`MetricType`](../../enums/metric_type.py) | JaccardSimilarity | What metrics should be calculated |
| `joint-model` | `bool` | False | If a joint model should be used instead of a single one |
| `joint-model-amount` | `int` | 2 | How many models should be trained jointly |
| `enable-external-logging` | `bool` | False | Should logging to external service be enabled |
| `train-dataset-limit-size` | `int` | None | Limit the train dataset |
| `validation-dataset-limit-size` | `int` | None | Limit the validation dataset |
| `skip-validation` | `bool` | False | Whether validation should be skipped, meaning no validation dataset is loaded and no evaluation is done while training |
| `run-experiments` | `bool` | False | Whether to run experiments instead of training or evaluation |
| `experiment-types` | [`ExperimentType`](../../enums/experiment_type.py) | None | What types of experiments should be run |
| `reset-training-on-early-stop` | `bool` | False | Whether resetting of training should be done if early stopping is activated and the first epoch has not yet been finished |
| `resets-limit` | `int` | 1 | How many times should the training be reset during first epoch if early stopping is activated |
| `training-reset-epoch-limit` | `int` | 1 | Until which epoch the training reset should be performed |
| `save-checkpoint-on-crash` | `bool` | False | If this is set to true, then in the event of an exception or crash of the program, the model's checkpoint will be saved to the file system |
| `save-checkpoint-on-finish` | `bool` | False | If this is set to true, then when the model has converged, its checkpoint will be saved to the file system. Keep in mind that this will not be the best model checkpoint as the stopping will occur after some amount of iterations without any improvement |

Additionally, the following arguments infer from [`PretrainedArgumentsService`](../../services/arguments/pretrained_arguments_service.py) and are used currently in all implemented challenge argument services

| Parameter     | Type          | Default value  | Description |
| ------------- | ------------- | -------------- |-------------|
| `pretrained-weights` | `str` | bert-base-cased | Weights to use for initializing HuggingFace transformer models |
| `include-pretrained-model` | `bool` | False | Should a pretrained model be used to provide more information |
| `pretrained-model-size` | `int` | 768 | The hidden size dimension of the pretrained model |
| `pretrained-max-length` | `int` | None | The maximum length the pretrained model (if any) |
| `learn-new-embeddings` | `bool` | False | Whether new embeddings should be learned next to the pretrained representation |
| `fasttext-model` | `str` | None | fasttext model to use for loading additional information |
| `include-fasttext-model` | `bool` | False | Should a fasttext model be used to provide more information |
| `fasttext-model-size` | `int` | 300 | The hidden size dimension of the fasttext model |
| `pretrained-model` | [`PretrainedModel`](../../enums/pretrained_model.py) | BERT | Pretrained model that will be used to tokenize strings and generate embeddings |
| `fine-tune-pretrained` | `bool` | False | If true, the loaded pre-trained model will be fine-tuned instead of being frozen |
| `fine-tune-after-convergence` | `bool` | False | If true, the loaded pre-trained model will be fine-tuned but only once the full model has converged |
| `fine-tune-learning-rate` | `float` | None | Different learning rate to use for pre-trained model. If None is given, then the global learning rate will be used |