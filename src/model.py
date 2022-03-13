from typing import Union
from abc import ABC, abstractmethod
from src.data import Data, DataType
from matplotlib.figure import Figure
from src.visualization import Visualization


class Model(ABC):
    """
    An abstract class used to define the schema of each module/model
    """
    visuals = {}

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

    @abstractmethod
    def visualize(self, key: DataType, dir_path: str):
        pass

    def _visualize(self, key: DataType, vis: Union[Visualization, Figure]):
        """
        Add Visualization object
        :param key: type of visualization
        :param vis: Visualization object
        """
        if key not in self.visuals:
            self.visuals[key] = []
        self.visuals[key].append(vis)
