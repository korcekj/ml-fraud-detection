from abc import ABC, abstractmethod

from graphviz import Source
from matplotlib import pyplot as plt
from matplotlib.figure import Figure as MatPlotFigure
from plotly.graph_objects import Figure as PlotlyFigure
from plotly.subplots import make_subplots
from sklearn.tree import export_graphviz, BaseDecisionTree

from src.utils import IO, Counter


class Visual(ABC):
    """
    A class used to represent a Visual object
    """

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def export(self, file_out: str, overwrite: bool):
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class Visualization:
    """
    A class used to represent a visualization process
    """

    def __init__(self):
        """
        Initialize object
        """
        self.__visuals = []

    def visualize(self, dir_path: str):
        """
        Show or export Visualization object
        :param dir_path: path to directory
        """
        if dir_path and not IO.is_dir(dir_path):
            raise Exception('Folder does not exist')

        for _ in range(len(self.__visuals)):
            visual = self.__visuals.pop(0)
            if dir_path is None:
                visual.show()
            else:
                visual.export(f'{dir_path}/{visual.name}.{Counter.next()}')

    def _visualize(self, vis: Visual):
        """
        Add Visualization object to the list of visualizations
        :param vis: Visualization object
        """
        self.__visuals.append(vis)


class MatPlotVis(Visual):
    """
    A class used to visualize a Graph object
    """

    def __init__(self, name: str, fig_size: (float, float) = None, rows: int = 1, cols: int = 1):
        """
        Initialize object
        :param name: name of the Visual
        :param fig_size: size of the figure
        :param rows: number of rows
        :param cols: number of columns
        """
        self.__name = name
        self.__rows = rows
        self.__cols = cols
        self.__fig_size = fig_size
        self.__figure = plt.figure(figsize=self.__fig_size)

    def show(self):
        """
        Display the graph
        :return Visual object
        """
        self.__figure.tight_layout()
        self.__figure.show()
        return self

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the Visual object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: Visual object
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        self.__figure.tight_layout()
        self.__figure.savefig(f'{file_out}.png')
        return self

    def add_graph(self, graph, x_lab: str = '', y_lab: str = '', position: int = 1):
        """
        Add graph into the existing layout
        :param graph: callback function returning graph
        :param x_lab: label on x
        :param y_lab: label on y
        :param position: position on the grid
        :return: Axes subclass object
        """
        ax = self.__figure.add_subplot(self.__rows, self.__cols, position)
        ax = graph(ax)
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        return ax

    @property
    def name(self):
        """
        Get name property
        :return: Visual name
        """
        return self.__name

    @property
    def figure(self):
        """
        Get figure property
        :return: Figure object
        """
        return self.__figure

    @figure.setter
    def figure(self, figure: MatPlotFigure):
        """
        Set figure property
        :param figure: Figure object
        """
        self.__figure = figure


class PlotlyVis(Visual):
    """
    A class used to visualize a Graph object
    """

    def __init__(self, name: str, titles: list = None, rows: int = 1, cols: int = 1):
        """
        Initialize object
        :param name: name of the Visual
        :param titles: list of graph titles
        :param rows: number of rows
        :param cols: number of columns
        """
        self.__name = name
        self.__rows = rows
        self.__cols = cols
        self.__titles = titles
        self.__figure = make_subplots(rows=self.__rows, cols=self.__cols, subplot_titles=self.__titles)

    def show(self):
        """
        Display the graph
        :return Visual object
        """
        self.__figure.show()
        return self

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the Visual object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: Visual object
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
        :return: Visual object
        """
        self.__figure.add_trace(graph, row=row, col=col)
        self.__figure.update_xaxes(title_text=x_lab, row=row, col=col)
        self.__figure.update_yaxes(title_text=y_lab, row=row, col=col)
        return self

    @property
    def name(self):
        """
        Get name property
        :return: Visual name
        """
        return self.__name

    @property
    def figure(self):
        """
        Get figure property
        :return: Figure object
        """
        return self.__figure

    @figure.setter
    def figure(self, figure: PlotlyFigure):
        """
        Set figure property
        :param figure: Figure object
        """
        self.__figure = figure


class TreeVis(Visual):
    """
    A class used to visualize a BaseDecisionTree object
    """

    def __init__(self, name: str, tree: BaseDecisionTree = None, features: list = None, classes: list = None):
        """
        Initialize object
        :param name: name of the Visual
        :param tree: BaseDecisionTree module/model
        """
        self.__name = name
        self.__tree = tree
        self.__features = features
        self.__classes = classes

    def show(self):
        """
        Display the graph
        """
        pass

    def export(self, file_out: str, overwrite: bool = False):
        """
        Export the Visual object to output file
        :param file_out: path to output file
        :param overwrite: boolean
        :return: Visual object
        """
        if not file_out:
            raise Exception('Output path is missing')

        if IO.is_file(file_out) and not overwrite:
            raise Exception('File already exists')

        dot_data = export_graphviz(self.__tree, filled=True, feature_names=self.__features, class_names=self.__classes)
        graph = Source(dot_data, format='png')
        graph.render(file_out, cleanup=True)
        return self

    @property
    def name(self):
        """
        Get name property
        :return: Visual name
        """
        return self.__name

    @property
    def tree(self):
        """
        Get tree property
        :return: BaseDecisionTree object
        """
        return self.__tree

    @tree.setter
    def tree(self, tree: BaseDecisionTree):
        """
        Set tree property
        :param tree: BaseDecisionTree object
        """
        self.__tree = tree
