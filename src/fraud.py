import os
import json
import datetime as dt
from src.data import Data
from src.logger import Logger
from math import isnan
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.parse import urlencode
from aiohttp import ClientSession
from asyncio import gather, run
from pandas import DataFrame, Series

load_dotenv()
MS_DISTANCE_URL = os.getenv("MS_DISTANCE_URL")

logger = Logger.get_logger()


def is_nan(variables: list) -> bool:
    def validate(var) -> bool:
        try:
            return isnan(float(var))
        except ValueError:
            return False

    return any(validate(var) for var in variables)


def date_diff_in_seconds(date1: str, date2: str, date_format: str = '%Y-%m-%d %H:%M:%S') -> int:
    if is_nan([date1]) or is_nan([date2]):
        return -1
    d1 = dt.datetime.strptime(date1, date_format)
    d2 = dt.datetime.strptime(date2, date_format)
    return int((d1 - d2).total_seconds())


def duration_diff_in_seconds_sync(address1: str, address2: str) -> int:
    if is_nan(address1.split(' ')) or is_nan(address2.split(' ')):
        return -1
    url_query = urlencode({'from': address1, 'to': address2})
    url = MS_DISTANCE_URL + url_query
    response = urlopen(url).read()
    json_data = json.loads(response)
    return json_data['duration']


async def duration_diff_in_seconds_async(address1: str, address2: str) -> int:
    if is_nan(address1.split(' ')) or is_nan(address2.split(' ')):
        return -1
    async with ClientSession() as session:
        url_query = urlencode({'from': address1, 'to': address2})
        url = MS_DISTANCE_URL + url_query
        response = await session.get(url)
        if response.status != 200:
            response.raise_for_status()
        json_data = await response.json()
        return json_data['duration']


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
        if i > 2:
            break
        transactions = data.get_df().loc[data.get_df()['cc_num'] == card]
        coroutines.append(process_transactions(transactions, i + 1, len(cards)))
    return coroutines


def find_fraudulent(data: Data, export_to: str = None):
    dataframes = run(run_coroutines(data))
    for dataframe in dataframes:
        data.merge(dataframe)
    if export_to is not None:
        data.export(export_to, True)
