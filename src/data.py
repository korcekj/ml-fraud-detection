import os
import math
import pandas as pd
from enum import Enum
from typing import Union, List
from src.utils import Singleton
from src.visualization import Visualization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from plotly import graph_objects as go
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor


class Scaler(StandardScaler, metaclass=Singleton):
    """
    A class used to represent a StandardScaler by Singleton pattern
    """

    def __init__(self):
        super().__init__()


class TorchDataset(Dataset):
    """
    A class used to represent a Torch dataset
    """

    def __init__(self, x_data: FloatTensor, y_data: FloatTensor):
        """
        :param x_data: tensor of features
        :param y_data: tensor of target
        """
        self.X = x_data
        self.y = y_data

    def __getitem__(self, index):
        """
        Get item based on the index
        :param index: row index in the dataset
        :return: tuple of tensors
        """
        return self.X[index], self.y[index]

    def __len__(self):
        """
        Get length of the feature tensor
        :return: length of the feature tensor
        """
        return len(self.X)


class DataType(Enum):
    """
    An enumeration used to represent type of the dataset
    """
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class Data:
    """
    A class used to represent a Pandas Dataframe with additional methods
    """

    def __init__(self, file_path: Union[str, None], df: Union[pd.DataFrame, None], dt: DataType, target: str):
        """
        :param file_path: path to the dataset
        :param df: DataFrame object
        :param dt: data type of the dataset
        :param target: column name
        """
        self.__file_path = file_path
        self.__df = df
        self.__dt = dt
        self.__target = target

        if self.__file_path is None and self.__df is None:
            raise Exception('Wrong initialization')

    @classmethod
    def file(cls, file_path: str, dt: DataType, target: str, rows: int = 0):
        """
        Initialize class from file
        :param file_path: path to the dataset
        :param dt: data type of the dataset
        :param target: column name
        :param rows: number of rows to pick up
        :return: Data object
        """
        data = cls(file_path, None, dt, target)
        data.load(rows)
        return data

    @classmethod
    def split_file(cls, file_path: str, dts: List[DataType], target: str, split_ratio: float = 0.2, rows: int = 0):
        """
        Initialize classes from file based on the split parameters
        :param file_path: path to the dataset
        :param dts: array of data types used in datasets
        :param target: column name
        :param split_ratio: ratio of "train" and "test" data
        :param rows: number of rows to pick up
        :return: Data objects
        """
        data_1 = cls(file_path, None, dts[0], target)
        data_1.load(rows)
        df_1, df_2 = train_test_split(data_1.get_df(), test_size=split_ratio, random_state=42)
        data_1.set_df(df_1)
        data_2 = cls(None, df_2, dts[1], target)
        return data_1, data_2

    @classmethod
    def dataframe(cls, df: pd.DataFrame, dt: DataType, target: str):
        """
        Initialize class from DataFrame object
        :param df: DataFrame object
        :param dt: data type of the dataset
        :param target: column name
        :return: Data object
        """
        data = cls(None, df, dt, target)
        return data

    @classmethod
    def split_dataframe(cls, df: pd.DataFrame, dts: List[DataType], target: str, split_ratio: int = 0.3):
        """
        Initialize classes from DataFrame based on the split parameters
        :param df: DataFrame object
        :param dts: array of data types used in datasets
        :param target: column name
        :param split_ratio: ratio of "train" and "test" data
        :return: Data objects
        """
        df_1, df_2 = train_test_split(df, test_size=split_ratio, random_state=42)
        data_1 = cls(None, df_1, dts[0], target)
        data_2 = cls(None, df_2, dts[1], target)
        return data_1, data_2

    def load(self, rows: int):
        """
        Load a dataset with certain number of rows or an entire table
        :param rows: number of rows to pick up
        """
        if rows > 0:
            self.__minify(self.__file_path, rows)
        else:
            self.__read(self.__file_path)

    def __minify(self, file_in: str, rows: int):
        """
        Read only a certain number of rows
        :param file_in: path to the dataset
        :param rows: number of rows to pick up
        """
        self.__read(file_in)
        self.__df = self.__df[:rows]

    def __read(self, file_in: str):
        """
        Read the dataset from input file
        :param file_in: path to the dataset
        """
        if not file_in:
            raise Exception('Input path is missing')

        if not os.path.isfile(file_in):
            raise Exception('Input file does not exist')

        self.__df = pd.read_csv(file_in)

    def __write(self, file_out: str, overwrite: bool):
        """
        Write the dataset to output file
        :param file_out: path to output file
        :param overwrite: boolean
        """
        if not file_out:
            raise Exception('Output path is missing')

        if os.path.isfile(file_out) and not overwrite:
            raise Exception('File already exists')

        self.__df.to_csv(file_out, index=False)

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the dataset to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: Data object
        """
        self.__write(file_out, overwrite)
        return self

    def merge(self, data):
        """
        Merge Data object
        :param data: Data object to be merged
        :return: Data object
        """
        self.__df = pd.concat([self.__df, data])
        return self

    def remove_null_cells(self):
        """
        Remove empty or null cells within the dataset
        :return: Data object
        """
        new_df: pd.DataFrame = self.__df.dropna()
        self.__df = new_df.reset_index(drop=True)
        return self

    def remove_columns(self, columns_to_remove: list):
        """
        Remove defined columns from the dataset
        :param columns_to_remove: list of column names to be removed
        :return: Data object
        """
        columns = [column for column in self.__df.columns if column in columns_to_remove]
        self.__df = self.__df.drop(columns=columns)
        return self

    def sort_columns(self, sort_by: dict):
        """
        Sort columns in the dataset
        :param sort_by: dictionary of column names and booleans as a value
        :return: Data object
        """
        new_df = self.__df.sort_values(by=list(sort_by.keys()), ascending=tuple(sort_by.values()))
        self.__df = new_df
        return self

    def encode(self):
        """
        Encode object columns within the dataset
        :return: Data object
        """
        new_df = self.__df.select_dtypes(include=['object']).astype('category')
        for column in new_df.columns:
            self.__df[column] = new_df[column].cat.codes
        return self

    def normalize(self):
        """
        Normalize feature columns in the dataset using a Scaler
        :return: Data object
        """
        if self.__target is None:
            raise Exception('Target is missing')

        scaler = Scaler()
        columns = self.get_features()

        if self.__dt == DataType.TRAIN:
            scaler.fit(self.__df[columns])

        scaled_values = scaler.transform(self.__df[columns])
        self.__df[columns] = pd.DataFrame(scaled_values, columns=columns)
        return self

    def get_dataset(self) -> Dataset:
        """
        Get a Torch dataset instance
        :return: TorchDataset object
        """
        if self.__target is None:
            raise Exception('Target is missing')
        return TorchDataset(
            x_data=FloatTensor(self.__df[self.get_features()].values),
            y_data=FloatTensor(self.__df[self.__target].values)
        )

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """
        Get a Torch dataloader instance
        :param batch_size: size of the batch as a number
        :param shuffle: boolean
        :return: DataLoader object
        """
        if self.__target is None:
            raise Exception('Target is missing')
        dataset = self.get_dataset()
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def get_features(self, target: bool = False) -> list:
        """
        Get feature columns
        :param target: boolean
        :return: list of columns
        """
        if target:
            return list(self.__df.columns)
        if self.__target is None:
            raise Exception('Target is missing')
        return [column for column in self.__df.columns if column != self.__target]

    def get_df(self) -> pd.DataFrame:
        """
        Get Pandas dataframe property
        :return: DataFrame object
        """
        return self.__df

    def set_df(self, df: pd.DataFrame):
        """
        Set Pandas dataframe property
        :param df: DataFrame object
        """
        self.__df = df

    def get_target(self) -> str:
        """
        Get target property
        :return: column name
        """
        return self.__target

    def set_target(self, target: str):
        """
        Set target property
        :param target: column name
        """
        self.__target = target

    def get_type(self) -> DataType:
        """
        Get data type property
        :return: type of the dataset
        """
        return self.__dt

    def set_type(self, dt: DataType):
        """
        Set data type property
        :param dt: DataType object
        """
        self.__dt = dt

    def vis_outliers(self):
        """
        Visualize outliers using the Visualization class
        """
        cols = 3
        rows = math.ceil(len(self.get_features(True)) / cols)
        vis = Visualization(titles=list(self.get_features(True)), rows=rows, cols=cols)
        col, row = 1, 1
        for column in self.get_features(True):
            if col > cols:
                row += 1
                col = 1
            vis.add_graph(go.Box(y=self.__df[column], name=column), row=row, col=col)
            col += 1
        vis.get_figure().update_layout(height=rows * 500, showlegend=False).show()

    def vis_correlation(self):
        """
        Visualize correlation with the dataset using the Visualization class
        """
        vis = Visualization()
        new_df = self.__df.corr()
        vis.add_graph(go.Heatmap(z=new_df, x=new_df.columns, y=new_df.columns))
        vis.show()

    def vis_target(self):
        """
        Visualize target ratio using the Visualization class
        """
        vis = Visualization(titles=['Number of frauds'])
        new_df = self.__df[self.__target]
        vis.add_graph(go.Bar(x=new_df.unique(), y=new_df.value_counts().values), x_lab='is_fraud', y_lab='count')
        vis.show()

    def print(self):
        """
        Print statistics about the dataset
        """
        def_cols = pd.get_option('display.max_columns')
        pd.set_option('display.max_columns', len(self.get_features(True)))
        print(f'\nDescription:\n{50 * "-"}')
        print(self.__df.describe(include='all'))
        print(f'\nInfo:\n{50 * "-"}')
        self.__df.info(verbose=True)
        pd.set_option('display.max_columns', def_cols)
