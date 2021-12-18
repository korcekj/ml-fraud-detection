import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from src.logger import Logger

logger = Logger.get_logger()


class BinaryClassification(nn.Module):
    def __init__(self, input_size: int):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batch_norm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


def accuracy(y_pred: torch.Tensor, y_test: torch.Tensor):
    # Round to 0 or 1
    y_pred = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return torch.round(acc * 100)


def train_model(model: nn.Module, device: torch.device, train_dl: DataLoader, lr: float = 0.001, epochs: int = 1):
    # Training mode
    model.train()
    # BCEWithLogitsLoss function automatically applies the Sigmoid activation in final layer during training
    criterion = nn.BCEWithLogitsLoss()
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            # Set gradient to 0 for each mini-batch
            optimizer.zero_grad()
            # Get a prediction
            y_pred = model(inputs)
            # Get the loss and accuracy
            loss = criterion(y_pred, targets.unsqueeze(1))
            acc = accuracy(y_pred, targets.unsqueeze(1))
            # Perform backpropagation
            loss.backward()
            optimizer.step()
            # Add all the mini-batch losses and accuracies
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        # Print average losses and accuracies for each epoch
        logger.info(
            f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_dl):.5f} | Acc: {epoch_acc / len(train_dl):.3f}'
        )


def evaluate_model(model: nn.Module, device: torch.device, test_dl: DataLoader):
    # Evaluation mode
    model.eval()
    y_pred_list = []
    y_actual_list = []
    # Do not perform back-propagation during inference
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            # Get a prediction
            y_pred = model(inputs)
            # Round to 0 or 1
            y_pred = torch.round(torch.sigmoid(y_pred))
            # Append to list of predictions and actuals
            y_pred_list.append(y_pred.cpu().numpy())
            y_actual_list.append(targets.cpu().numpy())

    # Flatten out the list
    y_pred_list = [x.squeeze().tolist() for x in y_pred_list]
    y_actual_list = [x.squeeze().tolist() for x in y_actual_list]

    # Classification report
    conf_matrix = confusion_matrix(y_actual_list, y_pred_list)
    class_report = classification_report(y_actual_list, y_pred_list)

    logger.info(f'\n{class_report}')
