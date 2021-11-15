from src.data import Data
from src.utils import date_diff_in_seconds, duration_diff_in_seconds


def validate_transaction(transaction):
    date_diff_sec = date_diff_in_seconds(
        transaction['trans_date_trans_time'],
        transaction['prev_trans_date_trans_time']
    )
    distance_diff_sec = duration_diff_in_seconds(
        f'{transaction["city"]} {transaction["state"]}',
        f'{transaction["prev_city"]} {transaction["prev_state"]}'
    )
    transaction['is_fraud_check'] = 1 if date_diff_sec <= distance_diff_sec else 0
    return transaction


def find_fraudulent(data: Data):
    df = data.get_df()
    cards = df['cc_num'].unique()
    for card in cards:
        transactions = df.loc[df['cc_num'] == card]
        transactions = transactions.sort_values(by='trans_date_trans_time', ascending=False)
        transactions['prev_trans_date_trans_time'] = transactions['trans_date_trans_time'].shift(-1)
        transactions['prev_city'] = transactions['city'].shift(-1)
        transactions['prev_state'] = transactions['state'].shift(-1)
        transactions = transactions.apply(validate_transaction, axis=1)
        transactions = transactions.loc[:, ~transactions.columns.str.startswith('prev_')]
        df = data.merge(transactions).get_df()
        break
    return


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
    train_data, test_data = load_data()
    prepare_data(train_data)
    prepare_data(test_data)
    find_fraudulent(train_data)
    return


if __name__ == '__main__':
    main()
