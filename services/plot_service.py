import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PlotService:
    def __init__(self):
        sns.set()

    def plot_histogram(
            self,
            values: list,
            number_of_bins: int = None,
            start_x: float = None,
            end_x: float = None,
            title: str = None):
        if not number_of_bins:
            number_of_bins = 10

        if not start_x:
            start_x = min(values)

        if not end_x:
            end_x = max(values)

        distance_bin = (end_x - start_x) / number_of_bins

        bins = np.arange(start_x, end_x, distance_bin)
        plt.hist(values, bins=bins, edgecolor='none')

        if title:
            plt.title(title)

        plt.show()
