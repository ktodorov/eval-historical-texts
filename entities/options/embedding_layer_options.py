from enums.embedding_type import EmbeddingType

from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions


class EmbeddingLayerOptions:
    def __init__(
            self,
            device: str,
            pretrained_representations_options: PretrainedRepresentationsOptions = None,
            learn_subword_embeddings: bool = False,
            include_pretrained_model: bool = False,
            merge_subword_embeddings: bool = False,
            vocabulary_size: int = None,
            subword_embeddings_size: int = None,
            learn_character_embeddings: int = False,
            character_embeddings_size: int = None,
            learn_word_embeddings: int = False,
            word_embeddings_size: int = None,
            pretrained_word_weights = None,
            output_embedding_type: EmbeddingType = EmbeddingType.SubWord,
            character_rnn_hidden_size: int = 64,
            dropout: float = 0.0,
            learn_manual_features: bool = False,
            manual_features_count: int = None):

        self.device = device

        self.include_pretrained_model = include_pretrained_model
        self.pretrained_representations_options = pretrained_representations_options

        self.vocabulary_size = vocabulary_size

        self.learn_subword_embeddings = learn_subword_embeddings
        self.merge_subword_embeddings = merge_subword_embeddings
        self.subword_embeddings_size = subword_embeddings_size

        self.learn_character_embeddings = learn_character_embeddings
        self.character_embeddings_size = character_embeddings_size
        self.character_rnn_hidden_size = character_rnn_hidden_size

        self.learn_word_embeddings = learn_word_embeddings
        self.word_embeddings_size = word_embeddings_size
        self.pretrained_word_weights = pretrained_word_weights

        self.output_embedding_type = output_embedding_type
        self.dropout = dropout

        self.learn_manual_features = learn_manual_features
        self.manual_features_count = manual_features_count
