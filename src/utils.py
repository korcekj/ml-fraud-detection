import os
import json
import datetime as dt
from dotenv import load_dotenv
from math import isnan
from urllib.request import urlopen
from urllib.parse import urlencode
from aiohttp import ClientSession

load_dotenv()
MS_DISTANCE_URL = os.getenv("MS_DISTANCE_URL")


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


def duration_diff_in_seconds(address1: str, address2: str) -> int:
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
