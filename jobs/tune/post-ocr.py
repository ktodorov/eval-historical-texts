import subprocess


def finetune_model(
        language: str,
        batch_size: int,
        patience: int,
        eval_frequency: int,
        pretrained_weights: str,
        include_pretr: bool,
        learning_rate: float,
        hidden_size: int,
        embedding_size: int,
        number_layers: int,
        dropout: float,
        finetune: bool,
        finetune_after_convergence: bool,
        finetune_lr: float,
        fast_text_model: str,
        include_fast_text: bool):
    assert not (finetune and finetune_after_convergence)

    arguments = [
        "bash",
        "jobs/start_job.sh",
        "jobs/train_char-to-char-encoder-decoder.sh",
        f"LANGUAGE={language}",
        f"BATCHSIZE={batch_size}",
        f"PATIENCE={patience}",
        f"EVALFREQ={eval_frequency}",
        f"PRETRAINEDWEIGHTS={pretrained_weights}",
        f"FASTTEXTMODEL={fast_text_model}",
        f"LR={learning_rate}",
        f"HIDDEN={hidden_size}",
        f"EMB={embedding_size}",
        f"LAYERS={number_layers}",
        f"DR={dropout}",
        f"FINETUNELR={finetune_lr}"]

    if include_pretr:
        arguments.append("INCLUDEPRETR=--include-pretrained-model")

    if include_fast_text:
        arguments.append("FASTTEXT=--include-fasttext-model")

    if finetune_after_convergence:
        arguments.append(f"FINETUNEAFTERCONVERGENCE=yes")

    if finetune:
        arguments.append(f"FINETUNE=yes")

    subprocess.run(arguments)


patience = 10
finetune_patience = 20
evaluation_frequency = 5000
batch_size = 8

language = 'english'
pretrained_weights = 'bert-base-cased'
fast_text_model = 'en-model-skipgram-300-minc5-ws5-maxn-6.bin'

# language='french'
# pretrained_weights='bert-base-multilingual-cased'
# fast_text_model='fr-model-skipgram-300minc20-ws5-maxn-6.bin'

# language='german'
# pretrained_weights='bert-base-german-cased'
# fast_text_model='de-model-skipgram-300-minc20-ws5-maxn-6.bin'

pretr_options = [True, False]
lr_options = [1e-4]
hidden_size_options = [256, 512]
embedding_size_options = [64, 128]
number_layers_options = [2, 3]
dropout_options = [0.8]
finetune_options = [(False, False), (True, False)]#, (False, True)]
finetune_lr_options = [1e-4]
fasttext_options = [False]#True, False]


for include_pretr in pretr_options:
    for learning_rate in lr_options:
        for hidden_size in hidden_size_options:
            for embedding_size in embedding_size_options:
                for number_layers in number_layers_options:
                    for dropout in dropout_options:
                        for (finetune, finetune_after_convergence) in finetune_options:
                            for finetune_lr in finetune_lr_options:
                                for include_fast_text in fasttext_options:
                                    # if dropout == 0.5 and number_layers == 2 and hidden_size == 512 and embedding_size == 64:
                                    #     continue

                                    current_patience = patience
                                    if finetune or finetune_after_convergence:
                                        if not include_pretr:
                                            continue

                                        current_patience = finetune_patience

                                    finetune_model(
                                        language=language,
                                        batch_size=batch_size,
                                        patience=current_patience,
                                        eval_frequency=evaluation_frequency,
                                        pretrained_weights=pretrained_weights,
                                        include_pretr=include_pretr,
                                        learning_rate=learning_rate,
                                        hidden_size=hidden_size,
                                        embedding_size=embedding_size,
                                        number_layers=number_layers,
                                        dropout=dropout,
                                        finetune=finetune,
                                        finetune_after_convergence=finetune_after_convergence,
                                        finetune_lr=finetune_lr,
                                        fast_text_model=fast_text_model,
                                        include_fast_text=include_fast_text)