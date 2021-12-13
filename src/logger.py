import os
import logging
from dotenv import load_dotenv

load_dotenv()
LOGGER_FORMAT = os.getenv("LOGGER_FORMAT")
DATE_FORMAT = os.getenv("DATE_FORMAT")


class Logger:
    __logger = logging.getLogger()

    @classmethod
    def get_logger(cls, level: int = logging.INFO):
        logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT)
        cls.__logger.setLevel(level)
        return cls.__logger
