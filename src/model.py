import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
from src.visualization import Visualization
from src.logger import Logger

# Initiate the logger
logger = Logger.get_logger()


class BinaryClassification(nn.Module):
    """
    A class used to represent a Torch module/model
    """

    def __init__(self, input_size: int):
        """
        :param input_size: number of feature columns
        """
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)

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


def get_model(*args) -> nn.Module:
    """
    Get Torch module/model
    :param args: arguments to be bypassed
    :return: Module object
    """
    model = BinaryClassification(*args)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def accuracy(y_pred: torch.Tensor, y_test: torch.Tensor) -> torch.Tensor:
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


def train_model(model: nn.Module, train_dl: DataLoader, lr: float = 0.001, epochs: int = 1):
    """
    Train Torch module/model
    :param model: Torch module/model
    :param train_dl: Torch training dataloader
    :param lr: learning rate
    :param epochs: number of epochs
    """
    # Enable training mode
    model.train()
    # Apply Binary Cross Entropy criterion function between the target and the input probabilities
    criterion = nn.BCELoss()
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Iterate over each epoch
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        # Iterate over each feature and label
        for inputs, targets in train_dl:
            # Transfer data to GPU if available
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            # Set gradient to 0 for each mini-batch
            optimizer.zero_grad()
            # Get a prediction
            y_pred = model(inputs)
            # Get the loss and accuracy
            loss = criterion(y_pred, targets.unsqueeze(1))
            acc = accuracy(y_pred, targets.unsqueeze(1))
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()
            # Add all the mini-batch losses and accuracies
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        # Print average losses and accuracies for each epoch
        logger.info(
            f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_dl):.5f} | Acc: {epoch_acc / len(train_dl):.3f}'
        )


def evaluate_model(model: nn.Module, test_dl: DataLoader):
    """
    Evaluate Torch module/model
    :param model: Torch module/model
    :param test_dl: Torch testing dataloader
    """
    # Enable evaluation mode
    model.eval()
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
            y_pred = model(inputs)
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
