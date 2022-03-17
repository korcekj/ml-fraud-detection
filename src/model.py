from abc import ABC, abstractmethod
from src.utils import IO
from src.data import Data, DataType
from src.visualization import Visualization


class Model(ABC):
    """
    An abstract class used to define the schema of each module/model
    """

    @classmethod
    @abstractmethod
    def create(cls, n_features: int, file_path: str = None):
        pass

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass

    @abstractmethod
    def export(self, dir_path: str, overwrite: bool):
        pass

    @abstractmethod
    def fit(self, train_data: Data, valid_data: Data, params: dict):
        pass

    @abstractmethod
    def evaluate(self, test_data: Data):
        pass
