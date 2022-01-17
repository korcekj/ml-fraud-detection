from time import perf_counter
from src.data import Data, DataType
from src.logger import Logger
from src.model import get_model, train_model, evaluate_model
from src.fraud import find_fraudulent

# Initiate the logger
logger = Logger.get_logger()


def clean_data(data: Data):
    """
    Clean the dataset from unwanted columns or empty cells
    :param data:
    """
    data.remove_columns(['Unnamed: 0'])
    data.remove_null_cells()


def prepare_data(data: Data):
    """
    Prepare data for further processing
    :param data: Data object
    """
    data.remove_columns(
        ['trans_date_trans_time', 'cc_num', 'first', 'last', 'trans_num', 'dob', 'merch_lat', 'merch_long', 'unix_time']
    )
    data.encode()
    data.normalize()


def main():
    """
    Main function to run
    """
    # Start timer
    time_start = perf_counter()

    # Load input data
    train_data = Data('data/fraudTrain.min.02.csv', DataType.TRAIN)
    train_data.set_target('is_fraud')
    test_data = Data('data/fraudTest.min.02.csv', DataType.TEST)
    test_data.set_target('is_fraud')

    # Clean input data
    clean_data(train_data)
    clean_data(test_data)

    # Find fraudulent card transactions
    # find_fraudulent(train_data, 'data/fraudTrain.out.min.csv')
    # find_fraudulent(test_data, 'data/fraudTest.out.min.csv')

    # Prepare data for further processing
    prepare_data(train_data)
    prepare_data(test_data)

    # Initialize data loaders
    train_dl = train_data.get_dataloader(shuffle=True)
    test_dl = test_data.get_dataloader(batch_size=1)

    # Initialize model
    features_size = len(train_data.get_features())
    model = get_model(features_size)

    # Train model
    train_model(model, train_dl, epochs=30, lr=0.001)

    # Evaluate model
    evaluate_model(model, test_dl)

    # Stop timer
    time_end = perf_counter()
    # Print the number of seconds it takes for the main function to run
    logger.info(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    main()
