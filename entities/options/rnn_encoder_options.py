from typing import Dict

from enums.entity_tag_type import EntityTagType
from entities.options.pretrained_representations_options import PretrainedRepresentationsOptions

class RNNEncoderOptions:
    def __init__(
            self,
            device: str,
            pretrained_representations_options: PretrainedRepresentationsOptions,
            number_of_tags: Dict[EntityTagType, int],
            use_attention: bool,
            learn_new_embeddings: bool,
            vocabulary_size: int,
            embeddings_size: int,
            dropout: float,
            hidden_dimension: int,
            bidirectional: bool,
            number_of_layers: int,
            merge_subword_embeddings: bool,
            learn_character_embeddings: bool,
            character_embeddings_size: int,
            character_hidden_size: int,
            learn_manual_features: bool = False,
            manual_features_count: int = None):

        self.device = device
        self.number_of_tags = number_of_tags
        self.use_attention = use_attention
        self.pretrained_representations_options = pretrained_representations_options
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
        self.character_hidden_size = character_hidden_size

        self.learn_manual_features = learn_manual_features
        self.manual_features_count = manual_features_count
