import os
import math
import pandas as pd
from src.visualization import Visualization
from sklearn.preprocessing import StandardScaler
from plotly import graph_objects as go
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor


class PandasDataset(Dataset):
    def __init__(self, x_data: FloatTensor, y_data: FloatTensor):
        self.X = x_data
        self.y = y_data

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class Data:
    def __init__(self, file_path: str, rows: int = 0):
        self.__file_path = file_path
        self.__df = pd.DataFrame()
        self.__target = None
        self.__load(rows)

    def __load(self, rows: int):
        if rows > 0:
            self.__minify(self.__file_path, rows)
        else:
            self.__read(self.__file_path)

    def __minify(self, file_in: str, rows: int):
        self.__read(file_in)
        self.__df = self.__df[:rows]

    def __read(self, file_in: str):
        if not file_in:
            raise Exception('Input path is missing')

        if not os.path.isfile(file_in):
            raise Exception('Input file does not exist')

        self.__df = pd.read_csv(file_in)

    def __write(self, file_out: str, overwrite: bool):
        if not file_out:
            raise Exception('Output path is missing')

        if os.path.isfile(file_out) and not overwrite:
            raise Exception('File already exists')

        self.__df.to_csv(file_out, index=False)

    def export(self, file_out: str, overwrite: bool = False):
        self.__write(file_out, overwrite)
        return self

    def merge(self, data):
        self.__df = pd.concat([self.__df, data])
        return self

    def remove_null_cells(self):
        new_df: pd.DataFrame = self.__df.dropna()
        self.__df = new_df.reset_index(drop=True)
        return self

    def remove_columns(self, columns_to_remove: list):
        columns = [column for column in self.__df.columns if column in columns_to_remove]
        self.__df = self.__df.drop(columns=columns)
        return self

    def sort_columns(self, sort_by: dict):
        new_df = self.__df.sort_values(by=list(sort_by.keys()), ascending=tuple(sort_by.values()))
        self.__df = new_df
        return self

    def encode(self):
        new_df = self.__df.select_dtypes(include=['object']).astype('category')
        for column in new_df.columns:
            self.__df[column] = new_df[column].cat.codes
        return self

    def normalize(self):
        if self.__target is None:
            raise Exception('Target is missing')
        scaler = StandardScaler()
        columns = self.get_features()
        scaled_values = scaler.fit_transform(self.__df[self.get_features()])
        self.__df[columns] = pd.DataFrame(scaled_values, columns=columns)
        return self

    def get_dataset(self) -> Dataset:
        if self.__target is None:
            raise Exception('Target is missing')
        return PandasDataset(
            x_data=FloatTensor(self.__df[self.get_features()].values),
            y_data=FloatTensor(self.__df[self.__target].values)
        )

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
        if self.__target is None:
            raise Exception('Target is missing')
        dataset = self.get_dataset()
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def get_features(self, target: bool = False) -> list:
        if target:
            return list(self.__df.columns)
        if self.__target is None:
            raise Exception('Target is missing')
        return [column for column in self.__df.columns if column != self.__target]

    def get_df(self) -> pd.DataFrame:
        return self.__df

    def set_df(self, df: pd.DataFrame):
        self.__df = df

    def set_target(self, target: str):
        self.__target = target

    def vis_outliers(self):
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
        vis = Visualization()
        new_df = self.__df.corr()
        vis.add_graph(go.Heatmap(z=new_df, x=new_df.columns, y=new_df.columns))
        vis.show()

    def vis_target(self):
        vis = Visualization(titles=['Number of frauds'])
        new_df = self.__df[self.__target]
        vis.add_graph(go.Bar(x=new_df.unique(), y=new_df.value_counts().values), x_lab='is_fraud', y_lab='count')
        vis.show()

    def print(self):
        def_cols = pd.get_option('display.max_columns')
        pd.set_option('display.max_columns', len(self.get_features(True)))
        print(f'\nDescription:\n{50 * "-"}')
        print(self.__df.describe(include='all'))
        print(f'\nInfo:\n{50 * "-"}')
        self.__df.info(verbose=True)
        pd.set_option('display.max_columns', def_cols)
