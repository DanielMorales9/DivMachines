from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    Base Classifier
    """

    @abstractmethod
    def _initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _init_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def _init_model(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, x, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    def set_params(self, **params):
        """
        Returns
        -------
        self
        """
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self


from divmachines.classifiers.mf import MF
from divmachines.classifiers.fm import FM
from divmachines.classifiers.fm.bpr import BPR_FM
from divmachines.classifiers.mf.bpr import BPR_MF
__all__ = ["FM", "MF", "Classifier", "BPR_FM", "BPR_MF"]