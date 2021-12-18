import torch
from time import perf_counter
from src.data import Data
from src.logger import Logger
from src.model import BinaryClassification, train_model, evaluate_model
from src.utils import find_fraudulent

logger = Logger.get_logger()


def clean_data(data: Data):
    data.remove_columns(['Unnamed: 0'])
    data.remove_null_cells()


def prepare_data(data: Data):
    data.remove_columns(
        ['trans_date_trans_time', 'cc_num', 'first', 'last', 'trans_num', 'dob', 'merch_lat', 'merch_long', 'unix_time']
    )
    data.encode()
    data.normalize()


def main():
    time_start = perf_counter()

    # Load input data
    train_data = Data('data/fraudTrain.min.csv')
    train_data.set_target('is_fraud')
    test_data = Data('data/fraudTest.min.csv')
    test_data.set_target('is_fraud')

    # Clean input data
    clean_data(test_data)
    clean_data(train_data)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification(12)
    model.to(device)

    # Train model
    train_model(model, device, train_dl, epochs=30)

    # Evaluate model
    evaluate_model(model, device, test_dl)

    time_end = perf_counter()
    logger.info(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    main()
