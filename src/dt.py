import click
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from plotly import graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
from src.visualization import Visualization
from src.data import Data, DataType
from src.model import Model
from src.utils import IO


class DecisionTree(Model):
    """
    A class used to represent a Scikit module/model
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.__model = DecisionTreeClassifier(max_features=n_features)

    @classmethod
    def create(cls, n_features: int, file_path: str = None):
        """
        Get Scikit module/model
        :param n_features: number of feature columns
        :param file_path: path to input file
        :return: Module object
        """
        model = cls(n_features)
        if file_path is not None:
            model.load(file_path)
        return model

    def info(self):
        """
        Get info about Scikit module/model
        :return: Module object
        """
        click.echo(self.__model)
        click.echo(self.__model.get_params())
        return self

    def visualize(self, key: DataType, dir_path: str):
        """
        Show or export Tree object
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
                try:
                    visual.savefig(f'{dir_path}/{key}.{index}.png')
                except AttributeError:
                    visual.export(f'{dir_path}/{key}.{index}.html')

    def __read(self, file_in: str):
        """
        Read the Scikit module/model from input file
        :param file_in: path to input file
        """
        if not file_in:
            raise Exception('Input path is missing')

        if not IO.is_file(file_in):
            raise Exception('Input file does not exist')

        self.__model = joblib.load(file_in)

    def __write(self, file_out: str, overwrite: bool):
        """
        Write the Scikit module/model to output file
        :param file_out: path to output file
        :param overwrite: boolean
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        joblib.dump(self.__model, file_out)

    def load(self, file_path: str):
        """
        Load a Scikit module/model
        :param file_path: path to input file
        :return: Module object
        """
        self.__read(file_path)
        return self

    def export(self, dir_path: str, overwrite: bool = False):
        """
        Export the Scikit module/model to output file
        :param dir_path: path to output folder
        :param overwrite: boolean
        :return: Module object
        """
        self.__write(f'{dir_path}/dt.model.joblib', overwrite)
        return self

    def fit(self, train_data: Data, valid_data: Data, params: dict):
        # Set model parameters
        self.__model.set_params(**params)
        # Prepare input and target data
        inputs = train_data.get_df()[train_data.get_features()]
        targets = train_data.get_df()[train_data.get_target()].values
        # Train the model
        self.__model.fit(inputs, targets)
        # Get training accuracy
        train_acc = self.__model.score(inputs, targets)
        # Print training accuracy for the whole model
        click.echo(
            f'[{datetime.now().strftime("%H:%M:%S")}] Training finished:'
            f' | Training Acc: {train_acc:.3f}'
        )

        # Visualize training results using the matplotlib library
        vis = plt.figure(figsize=(25, 20))
        plot_tree(self.__model, filled=True)
        self._visualize(DataType.TRAIN, vis)

    def evaluate(self, test_data: Data):
        # Prepare input and target data
        inputs = test_data.get_df()[test_data.get_features()]
        targets = test_data.get_df()[test_data.get_target()].values
        # Predict targets
        targets_predicted = self.__model.predict(inputs)
        # Classification report
        conf_matrix = confusion_matrix(targets, targets_predicted)
        class_report = classification_report(targets, targets_predicted)

        click.echo(f'\n{class_report}')

        # Visualize confusion matrix using the Visualization class
        vis = Visualization()
        vis.add_graph(go.Heatmap(z=conf_matrix, x=[0, 1], y=[0, 1]), x_lab='Predicted', y_lab='Actual')
        self._visualize(DataType.TEST, vis)
