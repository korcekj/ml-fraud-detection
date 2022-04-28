import os
from asyncio import gather, run
from datetime import datetime
from urllib.parse import urlencode

import click
from aiohttp import ClientSession
from pandas import DataFrame, Series

from src.data import Data
from src.utils import Validator


class Service:
    def __init__(self):
        self.__base_url = os.environ["MS_DISTANCE_URL"]

    def date_diff_in_seconds(self, date1: str, date2: str, date_format: str = '%Y-%m-%d %H:%M:%S') -> int:
        """
        Calculate the date difference in seconds
        :param date1: date value number 1
        :param date2: date value number 2
        :param date_format: format of the dates
        :return: date difference in seconds
        """
        if Validator.is_nan([date1]) or Validator.is_nan([date2]):
            return -1
        d1 = datetime.strptime(date1, date_format)
        d2 = datetime.strptime(date2, date_format)
        return int((d1 - d2).total_seconds())

    async def duration_diff_in_seconds(self, address1: str, address2: str) -> int:
        """
        Calculate the duration difference between places in seconds (asynchronously)
        :param address1: value representing an address number 1
        :param address2: value representing an address number 1
        :return: duration difference in seconds
        """
        if Validator.is_nan(address1.split(' ')) or Validator.is_nan(address2.split(' ')):
            return -1
        async with ClientSession() as session:
            url_query = urlencode({'from': address1, 'to': address2})
            url = self.__base_url + url_query
            response = await session.get(url)
            if response.status != 200:
                response.raise_for_status()
            json_data = await response.json()
            return json_data['duration']


class MicroServices:
    """
    A class used to represent a Microservices object
    """

    def __init__(self, data: Data):
        """
        Initialize object
        :param data: Data object
        """
        self.__data = data
        self.__service = Service()

    async def __validate_transaction(self, transaction: Series) -> int:
        """
        Validate the transaction based on the location parameter (asynchronously)
        :param transaction: transaction Series
        :return: 0 for a non-fraudulent transaction and 1 for a fraudulent transaction
        """
        try:
            date_diff_sec = self.__service.date_diff_in_seconds(
                transaction['trans_date_trans_time'],
                transaction['prev_trans_date_trans_time']
            )
            distance_diff_sec = await self.__service.duration_diff_in_seconds(
                f'{transaction["city"]} {transaction["state"]}',
                f'{transaction["prev_city"]} {transaction["prev_state"]}'
            )
            return 1 if date_diff_sec <= distance_diff_sec else 0
        except Exception as e:
            click.echo(f'[{datetime.now().strftime("%H:%M:%S")}] Transaction [{transaction["trans_num"]}]: {e}',
                       err=True)

    async def __client_transactions(self, transactions: DataFrame, target: str) -> DataFrame:
        """
        Process client transactions (asynchronously)
        :param transactions: transactions DataFrame
        :param target: column name
        :param index: transactions batch index
        :param total: total number of transactions
        :return: DataFrame object
        """
        transactions = transactions.sort_values(by='trans_date_trans_time', ascending=False)
        transactions['prev_trans_date_trans_time'] = transactions['trans_date_trans_time'].shift(-1)
        transactions['prev_city'] = transactions['city'].shift(-1)
        transactions['prev_state'] = transactions['state'].shift(-1)
        click.echo(f'[{datetime.now().strftime("%H:%M:%S")}] Card [{transactions["cc_num"].values[0]}]: Start')
        for row, transaction in transactions.iterrows():
            click.echo(f'[{datetime.now().strftime("%H:%M:%S")}] Transaction [{transaction["trans_num"]}]: Start')
            prev_is_fraud = transactions.loc[row, target]
            is_fraud = await self.__validate_transaction(transaction)
            transactions.loc[row, target] = prev_is_fraud if not is_fraud and prev_is_fraud else is_fraud
            click.echo(f'[{datetime.now().strftime("%H:%M:%S")}] Transaction [{transaction["trans_num"]}]: End')
        click.echo(f'[{datetime.now().strftime("%H:%M:%S")}] Card [{transactions["cc_num"].values[0]}]: End')
        return transactions.loc[:, ~transactions.columns.str.startswith('prev_')]

    async def __run(self):
        """
        Run coroutines using Python concurrency libraries (asynchronously)
        :return: list of processed transactions
        """
        transactions = []
        coroutines = self.coroutines
        for i in range(0, len(coroutines), 3):
            batch_coroutines = coroutines[i * 3: (i * 3) + 3]
            transactions += await gather(*batch_coroutines, return_exceptions=True)
        return transactions

    def fraudulent(self):
        """
        Find fraudulent transactions
        :return: MicroService object
        """
        transactions = run(self.__run())
        for transaction in transactions:
            self.__data.merge(transaction)
        return self

    @property
    def coroutines(self) -> list:
        """
        Generate transactions coroutines
        :return: list of coroutines generated
        """
        coroutines = []
        cards = self.__data.df['cc_num'].unique()
        target = self.__data.target
        for card in cards:
            transactions = self.__data.df.loc[self.__data.df['cc_num'] == card]
            coroutines.append(self.__client_transactions(transactions, target))
        return coroutines

    @property
    def data(self) -> Data:
        """
        Get Data object
        :return: Data object
        """
        return self.__data

    @data.setter
    def data(self, data: Data):
        """
        Set Data object
        :param data: Data object
        """
        self.__data = data
