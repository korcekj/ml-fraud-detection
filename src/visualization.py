from graphviz import Source
from src.utils import IO
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatFigure
from sklearn.tree import plot_tree, export_graphviz, BaseDecisionTree
from plotly.graph_objects import Figure as PltFigure
from plotly.subplots import make_subplots


class Visualization(ABC):
    """
    A class used to represent a visualization process
    """

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def export(self, file_out: str, overwrite: bool):
        pass


class PlotlyVis(Visualization):
    """
    A class used to visualize a Graph object
    """

    def __init__(self, figure: PltFigure = None, titles: list = None, rows: int = 1, cols: int = 1):
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
        Display the graph
        """
        self.__figure.show()
        return self

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the PlotlyVis object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: PlotlyVis object
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        self.__figure.write_html(f'{file_out}.html')
        return self

    def add_graph(self, graph, x_lab: str = '', y_lab: str = '', row: int = 1, col: int = 1):
        """
        Add graph into the existing layout
        :param graph: Plotly BaseTraceType object
        :param x_lab: label on x
        :param y_lab: label on y
        :param row: row index
        :param col: column index
        :return: PlotlyVis object
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

    def set_figure(self, figure: PltFigure):
        """
        Set figure property
        :param figure: Figure object
        :return: PlotlyVis object
        """
        self.__figure = figure
        return self


class TreeVis(Visualization):
    """
    A class used to visualize a BaseDecisionTree object
    """

    def __init__(self, tree: BaseDecisionTree = None):
        self.__tree = tree

    def show(self):
        """
        Display the graph
        """
        pass

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the TreeVis object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: TreeVis object
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        dot_data = export_graphviz(self.__tree, filled=True)
        graph = Source(dot_data, format='png')
        graph.render(file_out, cleanup=True)
        return self

    def set_tree(self, tree: BaseDecisionTree):
        """
        Set tree property
        :param tree: BaseDecisionTree object
        :return: TreeVis object
        """
        self.__tree = tree
        return self
