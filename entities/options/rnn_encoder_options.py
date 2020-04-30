from services.pretrained_representations_service import PretrainedRepresentationsService

class RNNEncoderOptions:
    def __init__(
            self,
            pretrained_representations_service: PretrainedRepresentationsService,
            device: str,
            number_of_tags: int,
            use_attention: bool,
            include_pretrained_model: bool,
            pretrained_model_size: int,
            include_fasttext_model: bool,
            fasttext_model_size: int,
            learn_new_embeddings: bool,
            vocabulary_size: int,
            embeddings_size: int,
            dropout: float,
            hidden_dimension: int,
            bidirectional: bool,
            number_of_layers: int,
            merge_subword_embeddings: bool,
            learn_character_embeddings: bool,
            character_embeddings_size: int):

        self.pretrained_representations_service = pretrained_representations_service
        self.device = device
        self.number_of_tags = number_of_tags
        self.use_attention = use_attention
        self.include_pretrained_model = include_pretrained_model
        self.pretrained_model_size = pretrained_model_size
        self.include_fasttext_model = include_fasttext_model
        self.fasttext_model_size = fasttext_model_size
        self.learn_new_embeddings = learn_new_embeddings
        self.vocabulary_size = vocabulary_size
        self.embeddings_size = embeddings_size
        self.dropout = dropout
        self.hidden_dimension = hidden_dimension
        self.bidirectional = bidirectional
        self.number_of_layers = number_of_layers
        self.merge_subword_embeddings = merge_subword_embeddings
        self.learn_character_embeddings = learn_character_embeddings
        self.character_embeddings_size = character_embeddings_size
