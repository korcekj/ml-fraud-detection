from time import perf_counter
from src.data import Data
from src.logger import Logger
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

    # Initialize datasets
    train_dataset = train_data.get_dataloader(shuffle=True)
    test_dataset = test_data.get_dataloader(batch_size=1)

    time_end = perf_counter()
    logger.info(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    main()
