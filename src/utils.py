import math
import os

from sklearn.preprocessing import StandardScaler


class Singleton(type):
    """
    A class used to represent a Singleton pattern
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Get instance of the Singleton object and create one if it does not exist
        :param args: list arguments
        :param kwargs: key value arguments
        :return: Singleton instance
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Scaler(StandardScaler, metaclass=Singleton):
    """
    A class used to represent a StandardScaler by Singleton pattern
    """

    def __init__(self):
        """
        Initialize object
        """
        super().__init__()


class Counter:
    """
    A static class used to represent a Counter
    """

    __state = 0

    @staticmethod
    def next():
        """
        Increment state by one
        :return: incremented state by one
        """
        Counter.__state += 1
        return Counter.__state


class Validator:
    """
    A static class used to represent a validation functionalities
    """

    @staticmethod
    def is_nan(variables: list) -> bool:
        """
        Check whether the provided list of variables contains NaNs
        :param variables: list of values to be checked
        :return: boolean
        """

        def validate(var) -> bool:
            """
            Validate the value against NaN check
            :param var: value to be checked
            :return: boolean
            """
            try:
                return math.isnan(float(var))
            except ValueError:
                return False

        return any(validate(var) for var in variables)


class IO:
    """
    A static class used to represent a IO functionalities
    """

    @staticmethod
    def is_file(file_path: str):
        """
        Check whether provided file exists
        :param file_path: path to the file
        :return: boolean
        """
        if file_path is None:
            return
        return os.path.isfile(file_path)

    @staticmethod
    def is_dir(dir_path: str):
        """
        Check whether provided directory exists
        :param dir_path: path to the directory
        :return: boolean
        """
        if dir_path is None:
            return
        return os.path.isdir(dir_path)

    @staticmethod
    def get_ext(file_path: str):
        """
        Get extension of the provided file
        :param file_path: path to the file
        :return: file extension
        """
        if file_path is None:
            return
        return os.path.splitext(file_path)[1]

    @staticmethod
    def create_dir(dir_root: str, dir_name: str):
        """
        Create directory
        :param dir_root: path to the root directory
        :param dir_name: name of a new directory
        :return: directory path
        """
        if dir_root is None or dir_name is None:
            return

        if not IO.is_dir(dir_root):
            return

        dir_path = f'{dir_root}/{dir_name}'
        if IO.is_dir(dir_path):
            return dir_path

        os.mkdir(dir_path)
        return dir_path

    @staticmethod
    def create_dirs(dir_path: str):
        """
        Create directories
        :param dir_path: path to a new directory/directories
        :return: directory path
        """
        if dir_path is None:
            return

        if IO.is_dir(dir_path):
            return dir_path

        os.makedirs(dir_path)
        return dir_path
