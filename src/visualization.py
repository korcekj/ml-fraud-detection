from src.utils import IO
from plotly import graph_objects as go
from plotly.subplots import make_subplots


class Visualization:
    """
    A class used to represent a Graph or set of Graphs
    """

    def __init__(self, figure: go.Figure = None, titles: list = None, rows: int = 1, cols: int = 1):
        """
        :param figure: Figure object
        :param titles: list of graph titles
        :param rows: number of rows
        :param cols: number of columns
        """
        self.__figure = figure
        self.__rows = rows
        self.__cols = cols
        self.__titles = titles
        self.__load()

    def __load(self):
        """
        Initiate graph by creating a layout
        """
        if self.__figure is None:
            self.__figure = make_subplots(rows=self.__rows, cols=self.__cols, subplot_titles=self.__titles)

    def show(self):
        """
        Display graph in a browser
        """
        self.__figure.show()

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the Visualization object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: Visualization object
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        self.__figure.write_html(file_out)
        return self

    def add_graph(self, graph, x_lab: str = '', y_lab: str = '', row: int = 1, col: int = 1):
        """
        Add graph into the existing layout
        :param graph: Plotly BaseTraceType object
        :param x_lab: label on x
        :param y_lab: label on y
        :param row: row index
        :param col: column index
        :return: Visualization object
        """
        self.__figure.add_trace(graph, row=row, col=col)
        self.__figure.update_xaxes(title_text=x_lab, row=row, col=col)
        self.__figure.update_yaxes(title_text=y_lab, row=row, col=col)
        return self

    def get_figure(self):
        """
        Get figure property
        :return: Figure object
        """
        return self.__figure

    def set_figure(self, figure: go.Figure):
        """
        Set figure property
        :param figure: Figure object
        :return: Figure object
        """
        self.__figure = figure
        return self
