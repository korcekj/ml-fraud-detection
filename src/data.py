import math
from enum import Enum
from typing import Optional, List

import click
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader

from src.utils import Scaler, IO
from src.visualization import Visualization, MatPlotVis


class TorchDataset(Dataset):
    """
    A class used to represent a Torch dataset
    """

    def __init__(self, x_data: FloatTensor, y_data: FloatTensor):
        """
        Initialize object
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

    def __str__(self):
        """
        Override default method to get a name of enum value
        :return: name of the value
        """
        return str(self.name)

    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    UNDEFINED = 4


class Data(Visualization):
    """
    A class used to represent a Pandas Dataframe with additional methods
    """

    def __init__(self, file_path: Optional[str], df: Optional[pd.DataFrame], dt: DataType, target: str):
        """
        Initialize object
        :param file_path: path to the dataset
        :param df: DataFrame object
        :param dt: data type of the dataset
        :param target: column name
        """
        super().__init__()
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
    def split_file(cls, file_path: str, dts: List[DataType], target: str, split_ratio: float = 0.3, rows: int = 0):
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
        df_1, df_2 = train_test_split(data_1.df, test_size=split_ratio, random_state=42)
        data_1.df = df_1
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
    def split_dataframe(cls, df: pd.DataFrame, dts: List[DataType], target: str, split_ratio: float = 0.3):
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

        if not IO.is_file(file_in):
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

        if IO.is_file(file_out) and not overwrite:
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
        if self.target is None:
            raise Exception('Target is missing')

        scaler = Scaler()
        columns = self.features

        if self.__dt == DataType.TRAIN:
            scaler.fit(self.__df[columns])

        scaled_values = scaler.transform(self.__df[columns])
        self.__df[columns] = pd.DataFrame(scaled_values, columns=columns)
        return self

    def balance(self, over_strategy: float = 0.1, under_strategy: float = 0.5):
        """
        Balance an imbalanced dataset
        :param over_strategy: ratio of over-sampled data
        :param under_strategy: ratio of under-sampled data
        :return: Data object
        """
        target = self.target
        columns = self.features
        over_sampler = SMOTE(sampling_strategy=over_strategy)
        under_sampler = RandomUnderSampler(sampling_strategy=under_strategy)
        steps = [('o', over_sampler), ('u', under_sampler)]
        pipeline = Pipeline(steps=steps)
        x_balanced, y_balanced = pipeline.fit_resample(self.__df[columns], self.__df[target])
        self.__df = pd.DataFrame(None)
        self.__df[columns] = pd.DataFrame(x_balanced, columns=columns)
        self.__df[target] = pd.DataFrame(y_balanced, columns=[target])
        return self

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """
        Get a Torch dataloader instance
        :param batch_size: size of the batch as a number
        :param shuffle: boolean
        :return: DataLoader object
        """
        if self.target is None:
            raise Exception('Target is missing')
        dataset = self.get_dataset()
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataset(self) -> Dataset:
        """
        Get a Torch dataset instance
        :return: TorchDataset object
        """
        if self.target is None:
            raise Exception('Target is missing')
        return TorchDataset(
            x_data=FloatTensor(self.__df[self.features].values),
            y_data=FloatTensor(self.__df[self.target].values)
        )

    @property
    def df(self) -> pd.DataFrame:
        """
        Get Pandas dataframe property
        :return: DataFrame object
        """
        return self.__df

    @df.setter
    def df(self, df: pd.DataFrame):
        """
        Set Pandas dataframe property
        :param df: DataFrame object
        """
        self.__df = df

    @property
    def columns(self):
        """
        Get DataFrame columns
        :return: list of columns
        """
        return self.df.columns

    @property
    def features(self) -> list:
        """
        Get feature columns
        :return: list of columns
        """
        if self.target is None:
            raise Exception('Target is missing')
        return [column for column in self.__df.columns if column != self.target]

    @property
    def target(self) -> str:
        """
        Get target property
        :return: column name
        """
        return self.__target

    @target.setter
    def target(self, target: str):
        """
        Set target property
        :param target: column name
        """
        self.target = target

    @property
    def type(self) -> DataType:
        """
        Get data type property
        :return: type of the dataset
        """
        return self.__dt

    @type.setter
    def type(self, dt: DataType):
        """
        Set data type property
        :param dt: DataType object
        """
        self.__dt = dt

    def vis_outliers(self):
        """
        Visualize outliers using the Visual class
        """
        cols = 3
        rows = math.ceil(len(self.columns) / cols)
        sns.set_theme()
        vis = MatPlotVis('outliers', rows=rows, cols=cols)
        for index, column in enumerate(self.columns, 1):
            vis.add_graph(
                lambda ax: sns.boxplot(x=self.__df[column], ax=ax),
                x_lab=column,
                position=index
            )
        self._visualize(vis)
        return self

    def vis_correlation(self):
        """
        Visualize correlation with the dataset using the Visual class
        :return: Data object
        """
        data = self.__df.corr()
        sns.set_theme()
        vis = MatPlotVis('correlation')
        vis.add_graph(lambda ax: sns.heatmap(data=data, ax=ax, annot=True, fmt='.1f', linewidths=.5))
        self._visualize(vis)
        return self

    def vis_target(self):
        """
        Visualize target ratio using the Visual class
        :return: Data object
        """
        data = self.__df.groupby(self.target)[self.target].count().to_frame()
        sns.set_theme()
        vis = MatPlotVis('target')
        vis.add_graph(
            lambda ax: sns.barplot(x=data.index, y=self.target, data=data, ax=ax),
            x_lab=self.target,
            y_lab='count'
        )
        self._visualize(vis)
        return self

    def info(self):
        """
        Print statistics about the dataset
        :return: Data object
        """
        def_cols = pd.get_option('display.max_columns')
        pd.set_option('display.max_columns', len(self.columns))
        click.echo(f'\nDescription:\n{50 * "-"}')
        click.echo(self.__df.describe(include='all'))
        click.echo(f'\nInfo:\n{50 * "-"}')
        self.__df.info(verbose=True)
        pd.set_option('display.max_columns', def_cols)
        return self
