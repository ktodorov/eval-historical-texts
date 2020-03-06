from services.data_service import DataService
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams['figure.dpi'] = 300


class PlotService:
    def __init__(
            self,
            data_service: DataService):
        sns.set()

        self._data_service = data_service

    def plot_histogram(
            self,
            values: list,
            number_of_bins: int = None,
            start_x: float = None,
            end_x: float = None,
            title: str = None,
            save_path: str = None,
            filename: str = None):
        if not number_of_bins:
            number_of_bins = 10

        if not start_x:
            start_x = min(values)

        if not end_x:
            end_x = max(values)

        distance_bin = (end_x - start_x) / number_of_bins

        bins = np.arange(start_x, end_x, distance_bin)

        plt.hist(values, bins=bins, edgecolor='none')

        self._add_properties(
            title,
            save_path,
            filename)


        if save_path is None or filename is None:
            plt.show()

        plt.clf()

    def plot_scatter(
            self,
            x_values: list,
            y_values: list,
            labels: list,
            title: str = None,
            save_path: str = None,
            filename: str = None):

        plt.scatter(x_values, y_values)

        plt.xlim(x_values.min()+0.00005, x_values.max()+0.00005)
        plt.ylim(y_values.min()+0.00005, y_values.max()+0.00005)

        for label, x, y in zip(labels, x_values, y_values):
            plt.annotate(label, xy=(x, y), xytext=(
                0, 0), textcoords='offset points')

        self._add_properties(
            title,
            save_path,
            filename)

        if save_path is None or filename is None:
            plt.show()

        plt.clf()

    def _add_properties(
            self,
            title: str = None,
            save_path: str = None,
            filename: str = None):

        if title is not None:
            plt.title(title)

        if save_path is not None and filename is not None:
            self._data_service.save_figure(save_path, filename, no_axis=False)
