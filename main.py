import sys
import click
from time import perf_counter
from src.data import Data, DataType
from src.model import NeuralNetwork
from src.fraud import find_fraudulent


@click.group()
def main():
    """
    Argument commands group
    """
    pass


@main.command('cd')
@click.option('-di', '--data-import', type=click.Path(exists=True), required=True, help='Data file path for import')
@click.option('-de', '--data-export', type=click.Path(), required=True, help='Data file path for export')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-r', '--rows', type=click.INT, default=0, help='Number of rows to be processed')
@click.option('-c', '--columns', help='Columns to be removed')
def clean_data(data_import: str, data_export: str, target: str, rows: int, columns: str):
    """
    Clean the dataset from unwanted columns and empty cells
    :param data_import: path to input file
    :param data_export: path to output file
    :param target: column name
    :param rows: number of rows to pick up
    :param columns: column names to be removed
    """
    # Start timer
    time_start = perf_counter()

    # Initialize data
    data = Data.file(data_import, DataType.UNDEFINED, target, rows)

    # Remove unnecessary data
    data.remove_null_cells()
    if columns is not None:
        data.remove_columns(columns.split(','))

    # Export data
    data.export(data_export, True)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


@main.command('fd')
@click.option('-di', '--data-import', type=click.Path(exists=True), required=True, help='Data file path for import')
@click.option('-de', '--data-export', type=click.Path(), required=True, help='Data file path for export')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-r', '--rows', type=click.INT, default=0, help='Number of rows to be processed')
def fraud_detection(data_import: str, data_export: str, target: str, rows: int):
    """
    Detect fraud transactions using microservice
    :param data_import: path to input file
    :param data_export: path to output file
    :param target: column name
    :param rows: number of rows to pick up
    """
    # Start timer
    time_start = perf_counter()

    # Initialize data
    data = Data.file(data_import, DataType.UNDEFINED, target, rows)

    # Find fraudulent transactions
    find_fraudulent(data)

    # Export data
    if data_export is not None:
        data.export(data_export, True)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


@main.command('nn')
@click.option('-tnd', '--train-data', type=click.Path(exists=True), required=True, help='Training data file path')
@click.option('-ttd', '--test-data', type=click.Path(exists=True), required=True, help='Testing data file path')
@click.option('-mi', '--module-import', type=click.Path(exists=True), help='Module file path for import')
@click.option('-me', '--module-export', type=click.Path(), help='Module folder path for export')
@click.option('-ve', '--visuals-export', type=click.Path(), help='Visualizations folder path for export')
@click.option('-vs', '--valid-split', type=click.FloatRange(0, 1), default=0.3, help='Validation split')
@click.option('-bs', '--batch-size', type=click.IntRange(1, 32_768), default=32, help='Batch size')
@click.option('-lr', '--learning-rate', type=click.FloatRange(0, 1), default=0.001, help='Learning rate')
@click.option('-e', '--epochs', type=click.IntRange(1, 10_000), default=100, help='Batch size')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-v', '--visuals', is_flag=True, help='Show visuals')
def neural_network(
        train_data: str,
        test_data: str,
        module_import: str,
        module_export: str,
        visuals_export: str,
        valid_split: float,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        target: str,
        visuals: bool
):
    """
    Detect fraud transactions using neural network
    :param train_data: path to training data
    :param test_data: path to testing data
    :param module_import: path to module for import
    :param module_export: path to module for export
    :param visuals_export: path to visuals folder
    :param valid_split: ratio of "valid" data
    :param batch_size: size of the batch
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param target: column name
    :param visuals: boolean
    """
    # Start timer
    time_start = perf_counter()

    # Load data
    data_train, data_valid = Data.split_file(train_data, [DataType.TRAIN, DataType.VALIDATION], target, valid_split)
    data_test = Data.file(test_data, DataType.TEST, target)

    # Normalize data
    data_train.remove_null_cells().encode().normalize()
    data_valid.remove_null_cells().encode().normalize()
    data_test.remove_null_cells().encode().normalize()

    # Initialize model
    features_size = len(data_train.get_features())
    model = NeuralNetwork.create(features_size, module_import).info()

    # Train model
    if module_import is None:
        model.fit(data_train, data_valid, batch_size, learning_rate, epochs)
        if visuals:
            model.visualize(DataType.TRAIN, visuals_export)

    # Evaluate model
    model.evaluate(data_test)
    if visuals:
        model.visualize(DataType.TEST, visuals_export)

    # Export model
    if module_export is not None:
        model.export(module_export, True)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    """
    Main function to run
    """
    args = sys.argv
    if "--help" in args or len(args) == 1:
        click.echo("Fraud detection")
    main()
