import sys
from time import perf_counter, time

import click

from src.data import Data, DataType
from src.dt import DecisionTree
from src.ktor import MicroServices
from src.nn import NeuralNetwork
from src.rf import RandomForest
from src.utils import IO


@click.group()
def main():
    """
    Argument commands group
    """
    pass


@main.command()
@click.option('-di', '--data-import', type=click.Path(exists=True), required=True, help='Data file path for import')
@click.option('-de', '--data-export', type=click.Path(), required=True, help='Data file path for export')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-r', '--rows', type=click.INT, default=0, help='Number of rows to be processed')
@click.option('-c', '--columns', help='Columns to be removed')
def data_cleanup(data_import: str, data_export: str, target: str, rows: int, columns: str):
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


@main.command()
@click.option('-di', '--data-import', type=click.Path(exists=True), required=True, help='Data file path for import')
@click.option('-de', '--data-export', type=click.Path(), required=True, help='Data file path for export')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-r', '--rows', type=click.INT, default=0, help='Number of rows to be processed')
def microservices(data_import: str, data_export: str, target: str, rows: int):
    """
    Detect fraud transactions using microservices
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
    ms = MicroServices(data).fraudulent()
    data = ms.data

    # Export data
    if data_export is not None:
        data.export(data_export, True)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


@main.command()
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
    batch_id = str(int(time()))

    # Load data
    data_train = Data.file(train_data, DataType.TRAIN, target)
    data_test = Data.file(test_data, DataType.TEST, target)

    # Normalize data
    data_train.remove_null_cells().encode().normalize()
    data_test.remove_null_cells().encode().normalize()

    # Split data
    data_train, data_valid = Data.split_dataframe(
        data_train.df,
        [DataType.TRAIN, DataType.VALIDATION],
        target,
        valid_split
    )

    # Visualize dataset
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        data_train.vis_target().vis_outliers().vis_correlation()
        data_train.visualize(dir_path)
        data_test.vis_target().vis_outliers().vis_correlation()
        data_test.visualize(dir_path)

    # Initialize model
    n_features = len(data_train.features)
    model = NeuralNetwork.create(n_features, module_import).info()

    # Train model
    if module_import is None:
        params = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        model.fit(data_train, data_valid, params)
        if visuals:
            dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
            model.visualize(dir_path)

    # Evaluate model
    model.evaluate(data_test)
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        model.visualize(dir_path)

    # Export model
    if module_export is not None:
        dir_path = IO.create_dirs(f'{module_export}/{batch_id}')
        model.export(dir_path)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


@main.command()
@click.option('-tnd', '--train-data', type=click.Path(exists=True), required=True, help='Training data file path')
@click.option('-ttd', '--test-data', type=click.Path(exists=True), required=True, help='Testing data file path')
@click.option('-mi', '--module-import', type=click.Path(exists=True), help='Module file path for import')
@click.option('-me', '--module-export', type=click.Path(), help='Module folder path for export')
@click.option('-ve', '--visuals-export', type=click.Path(), help='Visualizations folder path for export')
@click.option('-md', '--max-depth', type=click.INT, help='Maximum depth of the tree')
@click.option('-ms', '--min-samples-split', type=click.INT, default=2, help='Minimum number of samples to split a node')
@click.option('-ml', '--min-samples-leaf', type=click.INT, default=1, help='Minimum number of samples at a leaf node')
@click.option('-c', '--criterion', type=click.Choice(['gini', 'entropy']), default='gini', help='Quality function')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-v', '--visuals', is_flag=True, help='Show visuals')
def decision_tree(
        train_data: str,
        test_data: str,
        module_import: str,
        module_export: str,
        visuals_export: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        criterion: str,
        target: str,
        visuals: bool
):
    """
    Detect fraud transactions using decision tree
    :param train_data: path to training data
    :param test_data: path to testing data
    :param module_import: path to module for import
    :param module_export: path to module for export
    :param visuals_export: path to visuals folder
    :param max_depth: maximum depth of the tree
    :param min_samples_split: minimum number of samples to split a node
    :param min_samples_leaf: minimum number of samples at a leaf node
    :param criterion: split quality function
    :param target: column name
    :param visuals: boolean
    """
    # Start timer
    time_start = perf_counter()
    batch_id = str(int(time()))

    # Load data
    data_train = Data.file(train_data, DataType.TRAIN, target)
    data_test = Data.file(test_data, DataType.TEST, target)

    # Normalize data
    data_train.remove_null_cells().encode()
    data_test.remove_null_cells().encode()

    # Visualize dataset
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        data_train.vis_target().vis_outliers().vis_correlation()
        data_train.visualize(dir_path)
        data_test.vis_target().vis_outliers().vis_correlation()
        data_test.visualize(dir_path)

    # Initialize model
    n_features = len(data_train.features)
    model = DecisionTree.create(n_features, module_import).info()

    # Train model
    if module_import is None:
        params = {
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        model.fit(data_train, None, params)
        if visuals:
            dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
            model.visualize(dir_path)

    # Evaluate model
    model.evaluate(data_test)
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        model.visualize(dir_path)

    # Export model
    if module_export is not None:
        dir_path = IO.create_dirs(f'{module_export}/{batch_id}')
        model.export(dir_path)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the function to run
    click.echo(f'Task takes: {(time_end - time_start):.1f}s')


@main.command()
@click.option('-tnd', '--train-data', type=click.Path(exists=True), required=True, help='Training data file path')
@click.option('-ttd', '--test-data', type=click.Path(exists=True), required=True, help='Testing data file path')
@click.option('-mi', '--module-import', type=click.Path(exists=True), help='Module file path for import')
@click.option('-me', '--module-export', type=click.Path(), help='Module folder path for export')
@click.option('-ve', '--visuals-export', type=click.Path(), help='Visualizations folder path for export')
@click.option('-md', '--max-depth', type=click.INT, help='Maximum depth of the tree')
@click.option('-ms', '--min-samples-split', type=click.INT, default=2, help='Minimum number of samples to split a node')
@click.option('-ml', '--min-samples-leaf', type=click.INT, default=1, help='Minimum number of samples at a leaf node')
@click.option('-ne', '--n-estimators', type=click.INT, default=100, help='Number of trees')
@click.option('-c', '--criterion', type=click.Choice(['gini', 'entropy']), default='gini', help='Quality function')
@click.option('-t', '--target', required=True, help='Name of target column')
@click.option('-v', '--visuals', is_flag=True, help='Show visuals')
def random_forest(
        train_data: str,
        test_data: str,
        module_import: str,
        module_export: str,
        visuals_export: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        n_estimators: int,
        criterion: str,
        target: str,
        visuals: bool
):
    """
    Detect fraud transactions using random forest
    :param train_data: path to training data
    :param test_data: path to testing data
    :param module_import: path to module for import
    :param module_export: path to module for export
    :param visuals_export: path to visuals folder
    :param max_depth: maximum depth of the tree
    :param min_samples_split: minimum number of samples to split a node
    :param min_samples_leaf: minimum number of samples at a leaf node
    :param n_estimators: number of trees
    :param criterion: split quality function
    :param target: column name
    :param visuals: boolean
    """
    # Start timer
    time_start = perf_counter()
    batch_id = str(int(time()))

    # Load data
    data_train = Data.file(train_data, DataType.TRAIN, target)
    data_test = Data.file(test_data, DataType.TEST, target)

    # Normalize data
    data_train.remove_null_cells().encode()
    data_test.remove_null_cells().encode()

    # Visualize dataset
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        data_train.vis_target().vis_outliers().vis_correlation()
        data_train.visualize(dir_path)
        data_test.vis_target().vis_outliers().vis_correlation()
        data_test.visualize(dir_path)

    # Initialize model
    n_features = len(data_train.features)
    model = RandomForest.create(n_features, module_import).info()

    # Train model
    if module_import is None:
        params = {
            'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }
        model.fit(data_train, None, params)
        if visuals:
            dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
            model.visualize(dir_path)

    # Evaluate model
    model.evaluate(data_test)
    if visuals:
        dir_path = IO.create_dirs(f'{visuals_export}/{batch_id}')
        model.visualize(dir_path)

    # Export model
    if module_export is not None:
        dir_path = IO.create_dirs(f'{module_export}/{batch_id}')
        model.export(dir_path)

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
