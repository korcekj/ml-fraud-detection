import click
import joblib
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from src.visualization import Visualization, TreeVis, MatPlotVis
from src.data import Data
from src.model import Model
from src.utils import IO


class DecisionTree(Model, Visualization):
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
        inputs = train_data.df[train_data.features]
        targets = train_data.df[train_data.target].values
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
        vis = TreeVis('decision_tree_train', tree=self.__model)
        self._visualize(vis)

    def evaluate(self, test_data: Data):
        # Prepare input and target data
        inputs = test_data.df[test_data.features]
        targets = test_data.df[test_data.target].values
        # Predict targets
        targets_predicted = self.__model.predict(inputs)
        # Classification report
        conf_matrix = confusion_matrix(targets, targets_predicted)
        conf_matrix = conf_matrix / np.sum(conf_matrix)
        class_report = classification_report(targets, targets_predicted)

        click.echo(f'\n{class_report}')

        sns.set_theme()
        vis = MatPlotVis('decision_tree_test')
        vis.add_graph(
            lambda ax: sns.heatmap(data=conf_matrix, ax=ax, annot=True, cbar=False, fmt='.2%', linewidths=.5),
        )
        self._visualize(vis)
        return self
