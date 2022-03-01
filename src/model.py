import os
import torch
import torch.nn as nn
import torch.optim as optim
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
from src.data import Data
from src.logger import Logger
from src.visualization import Visualization

# Initiate the logger
logger = Logger.get_logger()


class NeuralNetwork(nn.Module):
    """
    A class used to represent a Torch module/model
    """

    def __init__(self, input_size: int):
        """
        :param input_size: number of feature columns
        """
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_size, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(32)

    @classmethod
    def create(cls, *args):
        """
        Get Torch module/model
        :param args: arguments to be bypassed
        :return: Module object
        """
        model = cls(*args)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
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

    def __read(self, file_in: str):
        """
        Read the Torch module/model from input file
        :param file_in: path to input file
        """
        if not file_in:
            raise Exception('Input path is missing')

        if not os.path.isfile(file_in):
            raise Exception('Input file does not exist')

        self.load_state_dict(torch.load(file_in))

    def __write(self, file_out: str, overwrite: bool):
        """
        Write the Torch module/model to output file
        :param file_out: path to output file
        :param overwrite: boolean
        """
        if not file_out:
            raise Exception('Output path is missing')

        if os.path.isfile(file_out) and not overwrite:
            raise Exception('File already exists')

        torch.save(self.state_dict(), file_out)

    def load(self, file_path: str):
        """
        Load a Torch module/model
        :param file_path: path to input file
        :return: Module object
        """
        self.__read(file_path)
        return self

    def export(self, file_path: str, overwrite: bool = False):
        """
        Export the Torch module/model to output file
        :param file_path: path to output file
        :param overwrite: boolean
        :return: Module object
        """
        self.__write(file_path, overwrite)
        return self

    def fit(
            self,
            train_data: Data,
            valid_data: Data,
            batch_size: int = 32,
            lr: float = 0.001,
            epochs: int = 1,
    ):
        """
        Train Torch module/model
        :param train_data: Pandas training dataframe
        :param valid_data: Pandas validation dataframe
        :param batch_size: size of the batch as a number
        :param lr: learning rate
        :param epochs: number of epochs
        """
        # Apply Binary Cross Entropy criterion function between the target and the input probabilities
        criterion = nn.BCELoss()
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # Initialize data loaders
        train_dl = train_data.get_dataloader(batch_size=batch_size, shuffle=True)
        valid_dl = valid_data.get_dataloader(batch_size=1)
        # Initialize result dictionaries
        train_results = {'loss': [], 'acc': []}
        valid_results = {'loss': [], 'acc': []}
        # Iterate over each epoch
        for epoch in range(1, epochs + 1):
            # Enable training mode
            self.train()
            train_loss = 0
            train_acc = 0
            # Iterate over each feature and label
            for inputs, targets in train_dl:
                # Transfer data to GPU if available
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
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
                    # Transfer data to GPU if available
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()
                    # Get a prediction
                    y_pred = self(inputs)
                    # Get the training loss and accuracy
                    loss = criterion(y_pred, targets.unsqueeze(1))
                    acc = self.__accuracy(y_pred, targets.unsqueeze(1))
                    # Add all the mini-batch losses and accuracies
                    valid_loss += loss.item()
                    valid_acc += acc.item()

            # Append average losses and accuracies for each epoch to list
            train_results['loss'].append(train_loss / len(train_dl))
            train_results['acc'].append(train_acc / len(train_dl))
            valid_results['loss'].append(valid_loss / len(valid_dl))
            valid_results['acc'].append(valid_acc / len(valid_dl))

            # Print average losses and accuracies for each epoch
            logger.info(
                f'Epoch {epoch:03}:'
                f' | Training Loss: {train_loss / len(train_dl):.5f}'
                f' | Training Acc: {train_acc / len(train_dl):.3f}'
                f' | Validation Loss: {valid_loss / len(valid_dl):.5f}'
                f' | Validation Acc: {valid_acc / len(valid_dl):.3f}'
            )

        # Visualize average losses and accuracies using the Visualization class
        vis = Visualization(rows=1, cols=2)
        vis.add_graph(go.Scatter(y=train_results['loss'], name='Train'), row=1, col=1, x_lab='Epoch', y_lab='Loss')
        vis.add_graph(go.Scatter(y=valid_results['loss'], name='Valid'), row=1, col=1, x_lab='Epoch', y_lab='Loss')
        vis.add_graph(go.Scatter(y=train_results['acc'], name='Train'), row=1, col=2, x_lab='Epoch', y_lab='Accuracy')
        vis.add_graph(go.Scatter(y=valid_results['acc'], name='Valid'), row=1, col=2, x_lab='Epoch', y_lab='Accuracy')
        vis.show()

    def evaluate(self, test_data: Data):
        """
        Evaluate Torch module/model
        :param test_data: Pandas testing dataframe
        """
        # Enable evaluation mode
        self.eval()
        # Initialize data loaders
        test_dl = test_data.get_dataloader(batch_size=1)
        # Initialize result arrays
        y_pred_list = []
        y_actual_list = []
        # Do not perform back-propagation during inference
        with torch.no_grad():
            # Iterate over each feature and label
            for inputs, targets in test_dl:
                # Transfer data to GPU if available
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
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

        # Classification report
        conf_matrix = confusion_matrix(y_actual_list, y_pred_list)
        class_report = classification_report(y_actual_list, y_pred_list)

        logger.info(f'\n{class_report}')

        # Visualize confusion matrix using the Visualization class
        vis = Visualization()
        vis.add_graph(go.Heatmap(z=conf_matrix, x=[0, 1], y=[0, 1]), x_lab='Predicted', y_lab='Actual')
        vis.show()
