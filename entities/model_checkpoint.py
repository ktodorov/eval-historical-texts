from datetime import datetime


class ModelCheckpoint:
    def __init__(
            self,
            model_dict: dict,
            epoch: int,
            iteration: int,
            best_metrics: object):

        self._model_dict = model_dict
        self._epoch = epoch
        self._iteration = iteration
        self._best_metrics = best_metrics
        self._date_saved = datetime.now()

    @property
    def model_dict(self) -> dict:
        return self._model_dict

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def best_metrics(self) -> object:
        return self._best_metrics

    @property
    def date_saved(self) -> datetime:
        return self._date_saved
