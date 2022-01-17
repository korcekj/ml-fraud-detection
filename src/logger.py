import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
LOGGER_FORMAT = os.getenv("LOGGER_FORMAT")
DATE_FORMAT = os.getenv("DATE_FORMAT")


class Logger:
    """
    A class used to represent a Logger
    """
    __logger = logging.getLogger()

    @classmethod
    def get_logger(cls, level: int = logging.INFO) -> logging.Logger:
        """
        Get logger instance statically
        :param level: level of logging
        :return: Logger object
        """
        logging.basicConfig(format=LOGGER_FORMAT, datefmt=DATE_FORMAT)
        cls.__logger.setLevel(level)
        return cls.__logger
