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
