import os
import logging
from time import perf_counter
from asyncio import gather, run
from pandas import DataFrame, Series
from src import Data, date_diff_in_seconds, duration_diff_in_seconds_async
from dotenv import load_dotenv

load_dotenv()
LOGGER_FORMAT = os.getenv("LOGGER_FORMAT")
DATE_FORMAT = os.getenv("DATE_FORMAT")

logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


async def validate_transaction(transaction: Series) -> int:
    try:
        date_diff_sec = date_diff_in_seconds(
            transaction['trans_date_trans_time'],
            transaction['prev_trans_date_trans_time']
        )
        distance_diff_sec = await duration_diff_in_seconds_async(
            f'{transaction["city"]} {transaction["state"]}',
            f'{transaction["prev_city"]} {transaction["prev_state"]}'
        )
        return 1 if date_diff_sec <= distance_diff_sec else 0
    except Exception as e:
        logger.exception(f'Transaction [{transaction["trans_num"]}]: {e}')


async def process_transactions(transactions: DataFrame, index: int, total: int) -> DataFrame:
    transactions = transactions.sort_values(by='trans_date_trans_time', ascending=False)
    transactions['prev_trans_date_trans_time'] = transactions['trans_date_trans_time'].shift(-1)
    transactions['prev_city'] = transactions['city'].shift(-1)
    transactions['prev_state'] = transactions['state'].shift(-1)
    logger.info(f'Card [{transactions["cc_num"].values[0]}]: Start')
    for row, transaction in transactions.iterrows():
        logger.info(f'Transaction [{transaction["trans_num"]}]: Start')
        transactions.loc[row, 'is_fraud_check'] = await validate_transaction(transaction)
        logger.info(f'Transaction [{transaction["trans_num"]}]: End')
    logger.info(f'Card [{transactions["cc_num"].values[0]}]: End')
    logger.info(f'--------------------{(index / total) * 100:.2f}%--------------------')
    return transactions.loc[:, ~transactions.columns.str.startswith('prev_')]


async def run_coroutines(data: Data):
    coroutines = generate_coroutines(data)
    return await gather(*coroutines, return_exceptions=True)


def generate_coroutines(data: Data) -> list:
    coroutines = []
    cards = data.get_df()['cc_num'].unique()
    for i, card in enumerate(cards):
        transactions = data.get_df().loc[data.get_df()['cc_num'] == card]
        coroutines.append(process_transactions(transactions, i + 1, len(cards)))
    return coroutines


def find_fraudulent(data: Data, export_to: str = None):
    dataframes = run(run_coroutines(data))
    for dataframe in dataframes:
        data.merge(dataframe)
    if export_to is not None:
        data.export(export_to, True)


def clean_data(data: Data):
    data.remove_null_cells()
    data.remove_columns(['Unnamed: 0'])


def load_data():
    # train_data = Data('data/fraudTrain.csv', 10000).export('data/fraudTrain.min.csv')
    # test_data = Data('data/fraudTest.csv', 10000).export('data/fraudTest.min.csv')
    train_data = Data('data/fraudTrain.min.csv')
    test_data = Data('data/fraudTest.min.csv')
    return train_data, test_data


def main():
    time_start = perf_counter()

    # Load input data
    train_data, test_data = load_data()

    # Clean input data
    clean_data(train_data)
    clean_data(test_data)

    # Find fraudulent card transactions
    # find_fraudulent(train_data, 'data/fraudTrain.out.min.csv')
    # find_fraudulent(test_data, 'data/fraudTest.out.min.csv')

    time_end = perf_counter()
    logger.info(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    main()
