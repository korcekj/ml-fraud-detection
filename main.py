from time import perf_counter
from asyncio import gather, run
from pandas import DataFrame, Series
from src.data import Data
from src.utils import date_diff_in_seconds, duration_diff_in_seconds_async


async def validate_transaction(transaction: Series) -> int:
    date_diff_sec = date_diff_in_seconds(
        transaction['trans_date_trans_time'],
        transaction['prev_trans_date_trans_time']
    )
    distance_diff_sec = await duration_diff_in_seconds_async(
        f'{transaction["city"]} {transaction["state"]}',
        f'{transaction["prev_city"]} {transaction["prev_state"]}'
    )
    return 1 if date_diff_sec <= distance_diff_sec else 0


async def process_transactions(transactions: DataFrame) -> DataFrame:
    transactions = transactions.sort_values(by='trans_date_trans_time', ascending=False)
    transactions['prev_trans_date_trans_time'] = transactions['trans_date_trans_time'].shift(-1)
    transactions['prev_city'] = transactions['city'].shift(-1)
    transactions['prev_state'] = transactions['state'].shift(-1)
    for row, transaction in transactions.iterrows():
        transactions.loc[row, 'is_fraud_check'] = await validate_transaction(transaction)
    return transactions.loc[:, ~transactions.columns.str.startswith('prev_')]


async def run_coroutines(data: Data):
    coroutines = generate_coroutines(data)
    return await gather(*coroutines)


def generate_coroutines(data: Data) -> list:
    coroutines = []
    cards = data.get_df()['cc_num'].unique()
    for card in enumerate(cards):
        transactions = data.get_df().loc[data.get_df()['cc_num'] == card]
        coroutines.append(process_transactions(transactions))
    return coroutines


def find_fraudulent(data: Data):
    dataframes = run(run_coroutines(data))
    for dataframe in dataframes:
        data.merge(dataframe)


def prepare_data(data: Data):
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
    train_data, test_data = load_data()
    prepare_data(train_data)
    prepare_data(test_data)
    find_fraudulent(train_data)
    find_fraudulent(test_data)
    time_end = perf_counter()
    print(f'Task takes: {(time_end - time_start):.1f}s')


if __name__ == '__main__':
    main()
