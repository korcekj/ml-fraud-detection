from abc import ABC, abstractmethod
from src.utils import IO
from src.data import Data, DataType
from src.visualization import Visualization


class Model(ABC):
    """
    An abstract class used to define the schema of each module/model
    """

    def __init__(self):
        self.visuals = {}

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass

    @abstractmethod
    def export(self, file_path: str, overwrite: bool):
        pass

    @abstractmethod
    def fit(self, train_data: Data, valid_data: Data, batch_size: int, lr: float, epochs: int):
        pass

    @abstractmethod
    def evaluate(self, test_data: Data):
        pass

    def _visualize(self, key: DataType, vis: Visualization):
        """
        Add Visualization object
        :param key: type of visualization
        :param vis: Visualization object
        """
        if key not in self.visuals:
            self.visuals[key] = []
        self.visuals[key].append(vis)

    def visualize(self, key: DataType, dir_path: str):
        """
        Show or export Visualization object
        :param key: type of visualization
        :param dir_path: path to directory
        """
        if key not in self.visuals:
            raise Exception('Key does not exists')

        if dir_path and not IO.is_dir(dir_path):
            raise Exception('Folder does not exist')

        for index, visual in enumerate(self.visuals[key], 1):
            if dir_path is None:
                visual.show()
            else:
                visual.export(f'{dir_path}/{key}.{index}.html')
