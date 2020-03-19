from services.data_service import DataService
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)


class PlotService:
    def __init__(
            self,
            data_service: DataService):
        sns.set()

        self._data_service = data_service

    def create_plot(self) -> matplotlib.axes.Axes:
        ax = plt.subplot()
        return ax

    def plot_histogram(
            self,
            values: list,
            number_of_bins: int = None,
            start_x: float = None,
            end_x: float = None,
            title: str = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        if not number_of_bins:
            number_of_bins = 10

        if not start_x:
            start_x = min(values)

        if not end_x:
            end_x = max(values)

        distance_bin = (end_x - start_x) / number_of_bins

        bins = np.arange(start_x, end_x, distance_bin)

        ax.hist(values, bins=bins, edgecolor='none')

        self._add_properties(
            ax,
            title,
            save_path,
            filename,
            hide_axis)

        if save_path is None or filename is None:
            plt.show()

        plt.clf()

    def plot_scatter(
            self,
            x_values: list,
            y_values: list,
            title: str = None,
            save_path: str = None,
            filename: str = None,
            color: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        ax.scatter(x_values, y_values, color=color)

        self._add_properties(
            ax,
            title,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

    def plot_labels(
            self,
            x_values: list,
            y_values: list,
            labels: list,
            color: str = None,
            title: str = None,
            save_path: str = None,
            filename: str = None,
            ax=None,
            show_plot: bool = True,
            bold_mask: list = None,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        for i, (label, x, y) in enumerate(zip(labels, x_values, y_values)):
            weight = 'light'
            if bold_mask is not None and bold_mask[i]:
                weight = 'bold'

            ax.annotate(label, xy=(x, y), xytext=(
                0, 0), textcoords='offset points', color=color, weight=weight)

        self._add_properties(
            ax,
            title,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

    def plot_arrow(
            self,
            x: float,
            y: float,
            dx: float,
            dy: float,
            title: str = None,
            save_path: str = None,
            filename: str = None,
            color: str = None,
            ax=None,
            show_plot: bool = True,
            hide_axis: bool = False):
        if ax is None:
            ax = self.create_plot()

        ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color),
                    bbox=dict(pad=7, facecolor="none", edgecolor="none"))

        self._add_properties(
            ax,
            title,
            save_path,
            filename,
            hide_axis)

        if show_plot and (save_path is None or filename is None):
            plt.show()

        if show_plot or (save_path is not None and filename is not None):
            plt.clf()

    def _add_properties(
            self,
            ax: matplotlib.axes.Axes,
            title: str = None,
            save_path: str = None,
            filename: str = None,
            hide_axis: bool = False):

        if hide_axis:
            ax.axis('off')

        if title is not None:
            plt.title(title)

        if save_path is not None and filename is not None:
            self._data_service.save_figure(save_path, filename, no_axis=False)
