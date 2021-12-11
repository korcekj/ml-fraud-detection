from plotly import graph_objects as go
from plotly.subplots import make_subplots


class Visualization:
    def __init__(self, figure: go.Figure = None, titles: list = None, rows: int = 1, cols: int = 1):
        self.__figure = figure
        self.__rows = rows
        self.__cols = cols
        self.__titles = titles
        self.__load()

    def __load(self):
        if self.__figure is None:
            self.__figure = make_subplots(rows=self.__rows, cols=self.__cols, subplot_titles=self.__titles)

    def add_graph(self, graph, x_lab: str = '', y_lab: str = '', row: int = 1, col: int = 1):
        self.__figure.add_trace(graph, row=row, col=col)
        self.__figure.update_xaxes(title_text=x_lab, row=row, col=col)
        self.__figure.update_yaxes(title_text=y_lab, row=row, col=col)
        return self

    def get_figure(self):
        return self.__figure

    def set_figure(self, figure: go.Figure):
        self.__figure = figure
        return self
