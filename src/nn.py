import click
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from src.visualization import Visualization, MatPlotVis
from src.utils import IO
from src.model import Model
from src.data import Data


class TorchNeuralNetwork(Module):
    """
    A class used to represent a Torch module
    """

    def __init__(self, n_features: int):
        """
        :param n_features: number of feature columns
        """
        super().__init__()
        self.layer_1 = nn.Linear(n_features, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate output of the network
        :param inputs: feature row of values
        :return: Tensor object
        """
        x = self.relu(self.layer_1(inputs))
        x = self.batch_norm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = torch.sigmoid(x)
        return x

    def __accuracy(self, y_pred: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
        """
        Calculate accuracy based on the prediction and the actual values
        :param y_pred: predicated label
        :param y_test: actual label
        :return: Tensor object
        """
        y_pred = torch.round(y_pred)
        correct_results_sum = (y_pred == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        return torch.round(acc * 100)

    def fit(self, train_dl: DataLoader, valid_dl: DataLoader, lr: float, epochs: int):
        """
        Train Torch module
        :param train_dl: Training dataloader
        :param valid_dl: Validation dataloader
        :param lr: learning rate
        :param epochs: number of epochs to run
        :return: tuple of losses and accuracies
        """
        # Apply Binary Cross Entropy criterion function between the target and the input probabilities
        criterion = nn.BCELoss()
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # Initialize result dictionaries
        loss_results = {'train': [], 'valid': []}
        acc_results = {'train': [], 'valid': []}
        # Iterate over each epoch
        for epoch in range(1, epochs + 1):
            # Enable training mode
            self.train()
            train_loss = 0
            train_acc = 0
            # Iterate over each feature and label
            for inputs, targets in train_dl:
                # Set gradient to 0 for each mini-batch
                optimizer.zero_grad()
                # Get a prediction
                y_pred = self(inputs)
                # Get the training loss and accuracy
                loss = criterion(y_pred, targets.unsqueeze(1))
                acc = self.__accuracy(y_pred, targets.unsqueeze(1))
                # Calculate gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # Add all the mini-batch losses and accuracies
                train_loss += loss.item()
                train_acc += acc.item()

            with torch.no_grad():
                # Enable evaluation mode
                self.eval()
                valid_loss = 0
                valid_acc = 0
                # Iterate over each feature and label
                for inputs, targets in valid_dl:
                    # Get a prediction
                    y_pred = self(inputs)
                    # Get the training loss and accuracy
                    loss = criterion(y_pred, targets.unsqueeze(1))
                    acc = self.__accuracy(y_pred, targets.unsqueeze(1))
                    # Add all the mini-batch losses and accuracies
                    valid_loss += loss.item()
                    valid_acc += acc.item()

            # Append average losses and accuracies for each epoch to list
            loss_results['train'].append(train_loss / len(train_dl))
            loss_results['valid'].append(valid_loss / len(valid_dl))
            acc_results['train'].append(train_acc / len(train_dl))
            acc_results['valid'].append(valid_acc / len(valid_dl))

            # Print average losses and accuracies for each epoch
            click.echo(
                f'[{datetime.now().strftime("%H:%M:%S")}] Epoch {epoch:03}:'
                f' | Training Loss: {loss_results["train"][-1]:.3f}'
                f' | Training Acc: {acc_results["train"][-1]:.3f}'
                f' | Validation Loss: {loss_results["valid"][-1]:.3f}'
                f' | Validation Acc: {acc_results["valid"][-1]:.3f}'
            )

        return loss_results, acc_results

    def evaluate(self, test_dl: DataLoader):
        """
        Evaluate Torch module
        :param test_dl: Testing dataloader
        :return: tuple of actual and predicated values
        """
        # Enable evaluation mode
        self.eval()
        # Initialize result arrays
        y_pred_list = []
        y_actual_list = []
        # Do not perform back-propagation during inference
        with torch.no_grad():
            # Iterate over each feature and label
            for inputs, targets in test_dl:
                # Get a prediction
                y_pred = self(inputs)
                # Get an actual
                actual = targets.numpy()
                # Get and round prediction to 0 or 1
                y_pred = y_pred.detach().numpy()
                y_pred = y_pred.round()
                # Append to list of predicted and actual values
                y_pred_list.append(y_pred)
                y_actual_list.append(actual)

        # Flatten out the lists
        y_pred_list = [x.squeeze().tolist() for x in y_pred_list]
        y_actual_list = [x.squeeze().tolist() for x in y_actual_list]

        return y_actual_list, y_pred_list


class NeuralNetwork(Model, Visualization):
    """
    A class used to represent a Model
    """

    def __init__(self, n_features: int):
        """
        :param n_features: number of feature columns
        """
        super().__init__()
        self.__model = TorchNeuralNetwork(n_features)

    @classmethod
    def create(cls, n_features: int, file_path: str = None):
        """
        Get Torch module/model
        :param n_features: number of feature columns
        :param file_path: path to input file
        :return: Model object
        """
        model = cls(n_features)
        if file_path is not None:
            model.load(file_path)
        return model

    def info(self):
        """
        Get info about Torch module/model
        :return: Model object
        """
        click.echo(self.__model)
        return self

    def __read(self, file_in: str):
        """
        Read the Torch module/model from input file
        :param file_in: path to input file
        """
        if not file_in:
            raise Exception('Input path is missing')

        if not IO.is_file(file_in):
            raise Exception('Input file does not exist')

        self.__model.load_state_dict(torch.load(file_in))

    def __write(self, file_out: str, overwrite: bool):
        """
        Write the Model object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        torch.save(self.__model.state_dict(), file_out)

    def load(self, file_path: str):
        """
        Load a Model object
        :param file_path: path to input file
        :return: Model object
        """
        self.__read(file_path)
        return self

    def export(self, dir_path: str, overwrite: bool = False):
        """
        Export the Model object to output file
        :param dir_path: path to output folder
        :param overwrite: boolean
        :return: Model object
        """
        self.__write(f'{dir_path}/nn.model.pth', overwrite)
        return self

    def fit(self, train_data: Data, valid_data: Data, params: dict):
        """
        Train Model
        :param train_data: Training data
        :param valid_data: Validation data
        :param params: size of the batch as a number
        :return: Model object
        """
        # Initialize data loaders
        train_dl = train_data.get_dataloader(batch_size=params['batch_size'], shuffle=True)
        valid_dl = valid_data.get_dataloader(batch_size=1)
        # Train the model
        loss, acc = self.__model.fit(train_dl, valid_dl, params['learning_rate'], params['epochs'])

        # Visualize training results using the matplotlib library
        sns.set_theme()
        vis = MatPlotVis('neural_network_train', rows=1, cols=2)
        vis.add_graph(
            lambda ax: sns.lineplot(data=loss, ax=ax),
            x_lab='epoch',
            y_lab='loss',
            position=1
        )
        vis.add_graph(
            lambda ax: sns.lineplot(data=acc, ax=ax),
            x_lab='epoch',
            y_lab='accuracy',
            position=2
        )
        self._visualize(vis)
        return self

    def evaluate(self, test_data: Data):
        """
        Evaluate Model
        :param test_data: Testing data
        :return: Model object
        """
        # Initialize data loaders
        test_dl = test_data.get_dataloader(batch_size=1)
        # Predict targets
        targets, targets_predicted = self.__model.evaluate(test_dl)
        # Classification report
        conf_matrix = confusion_matrix(targets, targets_predicted)
        conf_matrix = conf_matrix / np.sum(conf_matrix)
        class_report = classification_report(targets, targets_predicted)

        click.echo(f'\n{class_report}')

        sns.set_theme()
        vis = MatPlotVis('neural_network_test')
        vis.add_graph(
            lambda ax: sns.heatmap(data=conf_matrix, ax=ax, annot=True, cbar=False, fmt='.2%', linewidths=.5),
        )
        self._visualize(vis)
        return self
