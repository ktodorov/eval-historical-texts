from enums.pretrained_model import PretrainedModel

class PretrainedRepresentationsOptions:
    def __init__(
            self,
            include_pretrained_model: bool,
            include_fasttext_model: bool = False,
            pretrained_model_size: int = None,
            pretrained_weights: str = None,
            pretrained_max_length: int = None,
            pretrained_model: PretrainedModel = PretrainedModel.BERT,
            fine_tune_pretrained: bool = False,
            fine_tune_after_convergence: bool = False,
            fasttext_model: str = None,
            fasttext_model_size: int = None):
        self.include_pretrained_model = include_pretrained_model
        self.pretrained_model_size = pretrained_model_size
        self.pretrained_weights = pretrained_weights
        self.pretrained_max_length = pretrained_max_length
        self.pretrained_model = pretrained_model
        self.fine_tune_pretrained = fine_tune_pretrained
        self.fine_tune_after_convergence = fine_tune_after_convergence
        self.include_fasttext_model = include_fasttext_model
        self.fasttext_model = fasttext_model
        self.fasttext_model_size = fasttext_model_size

        assert not include_pretrained_model or (
            include_pretrained_model and pretrained_model_size is not None and pretrained_weights is not None and pretrained_max_length is not None)

        assert not include_fasttext_model or (
            include_fasttext_model and fasttext_model is not None and fasttext_model_size is not None)
