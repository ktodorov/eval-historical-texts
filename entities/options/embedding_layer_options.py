from services.pretrained_representations_service import PretrainedRepresentationsService
from enums.embedding_type import EmbeddingType

class EmbeddingLayerOptions:
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
            learn_subword_embeddings: bool = False,
            include_pretrained_model: bool = False,
            merge_subword_embeddings: bool = False,
            pretrained_model_size: int = None,
            include_fasttext_model: bool = False,
            fasttext_model_size: int = None,
            vocabulary_size: int = None,
            subword_embeddings_size: int = None,
            learn_character_embeddings: int = False,
            character_embeddings_size: int = None,
            output_embedding_type: EmbeddingType = EmbeddingType.SubWord,
            character_rnn_hidden_size: int = 64,
            dropout: float = 0.0):

        self.pretrained_representations_service = pretrained_representations_service
        self.device = device
        self.learn_subword_embeddings = learn_subword_embeddings
        self.include_pretrained_model = include_pretrained_model
        self.merge_subword_embeddings = merge_subword_embeddings
        self.pretrained_model_size = pretrained_model_size
        self.include_fasttext_model = include_fasttext_model
        self.fasttext_model_size = fasttext_model_size
        self.vocabulary_size = vocabulary_size
        self.subword_embeddings_size = subword_embeddings_size
        self.learn_character_embeddings = learn_character_embeddings
        self.character_embeddings_size = character_embeddings_size
        self.output_embedding_type = output_embedding_type
        self.character_rnn_hidden_size = character_rnn_hidden_size
        self.dropout = dropout
