import os
from sklearn.preprocessing import StandardScaler


class Singleton(type):
    """
    A class used to represent a Singleton pattern
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Scaler(StandardScaler, metaclass=Singleton):
    """
    A class used to represent a StandardScaler by Singleton pattern
    """

    def __init__(self):
        super().__init__()


class Counter:
    state = 0

    @staticmethod
    def next():
        Counter.state += 1
        return Counter.state


class IO:
    @staticmethod
    def is_file(file_path: str):
        if file_path is None:
            return
        return os.path.isfile(file_path)

    @staticmethod
    def is_dir(dir_path: str):
        if dir_path is None:
            return
        return os.path.isdir(dir_path)

    @staticmethod
    def get_ext(file_path: str):
        if file_path is None:
            return
        return os.path.splitext(file_path)[1]

    @staticmethod
    def create_dir(dir_root: str, dir_name: str):
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
        if dir_path is None:
            return

        if IO.is_dir(dir_path):
            return dir_path

        os.makedirs(dir_path)
        return dir_path
