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
        include_fast_text: bool,
        entity_tags: str,
        split_type: str,
        char_emb_size: int,
        char_hidden_size: int,
        no_new_embeddings: bool):
    assert not (finetune and finetune_after_convergence)


#     bash jobs/start_job.sh jobs/train_ner.sh PATIENCE=10 EF=20 BATCH=16 LANGUAGE='french' PRETRAINEDWEIGHTS='bert-base-multilingual-cased' FASTTEXT='--include-fasttext-model' FASTTEXTMODEL='fr-model-skipgram-300minc20-ws5-maxn-6.bin' PRETRAINEDMODEL='bert' NOATTENTION='--no-attention' INCLUDEPRETR='--include-pretrained-model' ENTITYTAGS='all' LR=1e-2 HIDDEN=256 EMB=64 LAYERS=1 DR=0.5 CHARESIZE=16 CHARHSIZE=32 NONEWEMBEDDINGS='yes' SPLIT='multi-seg'

    arguments = [
        "bash",
        "jobs/start_job.sh",
        "jobs/train_ner.sh",
        f"LANGUAGE={language}",
        f"BATCH={batch_size}",
        f"PATIENCE={patience}",
        f"EF={eval_frequency}",
        f"PRETRAINEDWEIGHTS={pretrained_weights}",
        f"FASTTEXTMODEL={fast_text_model}",
        f"LR={learning_rate}",
        f"HIDDEN={hidden_size}",
        f"EMB={embedding_size}",
        f"LAYERS={number_layers}",
        f"DR={dropout}",
        f"FINETUNELR={finetune_lr}",
        f"PRETRAINEDMODEL=bert",
        f"NOATTENTION=--no-attention",
        f"ENTITYTAGS={entity_tags}",
        f"SPLIT={split_type}"]

    if include_pretr:
        arguments.append("INCLUDEPRETR=--include-pretrained-model")

    if include_fast_text:
        arguments.append("FASTTEXT=--include-fasttext-model")

    if finetune_after_convergence:
        arguments.append(f"FINETUNEAFTERCONVERGENCE=yes")

    if finetune:
        arguments.append(f"FINETUNE=yes")

    if char_emb_size is not None and char_hidden_size is not None:
        arguments.append(f"CHARESIZE={char_emb_size}")
        arguments.append(f"CHARHSIZE={char_hidden_size}")

    if no_new_embeddings:
        arguments.append(f"NONEWEMBEDDINGS=yes")

    subprocess.run(arguments)


patience = 7
finetune_patience = 15
evaluation_frequency = 20
batch_size = 4

language = 'english'
pretrained_weights = 'bert-base-cased'
fast_text_model = 'en-model-skipgram-300-minc5-ws5-maxn-6.bin'

# language = 'french'
# pretrained_weights = 'bert-base-multilingual-cased'
# fast_text_model = 'fr-model-skipgram-300minc20-ws5-maxn-6.bin'

# language='german'
# pretrained_weights='bert-base-german-cased'
# fast_text_model='de-model-skipgram-300-minc20-ws5-maxn-6.bin'

pretr_options = [True, False]
lr_options = [1e-2]
hidden_size_options = [256]
embedding_size_options = [64]
number_layers_options = [1]
dropout_options = [0.8]
finetune_options = [
    (False, False),
    (True, False),
    (False, True)
]

finetune_lr_options = [1e-4]
fasttext_options = [True, False]

entity_tags_options = ['all', '1', '2']
split_type_options = ['seg']
char_size_options = [(16, 32)]
no_new_embeddings_options = [False, True]

for include_pretr in pretr_options:
    for include_fast_text in fasttext_options:
        for learning_rate in lr_options:
            for hidden_size in hidden_size_options:
                for embedding_size in embedding_size_options:
                    if embedding_size > 512:
                        if include_pretr and include_fast_text:
                            continue

                        embedding_size = 0
                        if not include_pretr:
                            embedding_size += 768

                        if not include_fast_text:
                            embedding_size += 300

                    for number_layers in number_layers_options:
                        for dropout in dropout_options:
                            for (finetune, finetune_after_convergence) in finetune_options:
                                batch_size = 4
                                current_patience = patience
                                if finetune or finetune_after_convergence:
                                    if not include_pretr:
                                        continue

                                    current_patience = finetune_patience
                                    batch_size = 2

                                for finetune_lr in finetune_lr_options:
                                    for entity_tags in entity_tags_options:
                                        for split_type in split_type_options:
                                            for (char_emb_size, char_hidden_size) in char_size_options:
                                                for no_new_embeddings in no_new_embeddings_options:
                                                    if no_new_embeddings and not include_pretr and not include_fast_text:
                                                        continue

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
                                                        include_fast_text=include_fast_text,
                                                        entity_tags=entity_tags,
                                                        split_type=split_type,
                                                        char_emb_size=char_emb_size,
                                                        char_hidden_size=char_hidden_size,
                                                        no_new_embeddings=no_new_embeddings)
