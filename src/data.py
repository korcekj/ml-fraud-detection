import pandas as pd
import os


class Data:
    def __init__(self, file_path: str, rows: int = 0):
        self.__file_path = file_path
        self.__df = pd.DataFrame()
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

    def get_df(self) -> pd.DataFrame:
        return self.__df

    def set_df(self, df: pd.DataFrame):
        self.__df = df