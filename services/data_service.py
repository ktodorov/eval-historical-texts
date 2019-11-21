import _pickle as pickle
import os
from collections import defaultdict
from datetime import datetime
import time
import logging

import matplotlib.pyplot as plt
from services.config_service import ConfigService


class DataService:

    def __init__(
            self,
            logger: logging.Logger):

        # determines relative disk directory for saving/loading
        self.stamp: str = ''
        self.actual_date: datetime = None

    def save_python_obj(self, obj: object, path: str, name: str, print_success: bool = True) -> bool:
        """ Saves python object to disk in pickle """

        try:
            filepath = os.path.join(path, f'{name}.pickle')
            with open(filepath, 'wb') as handle:
                pickle.dump(obj, handle, protocol=-1)

                if (print_success):
                    print("Saved {}".format(name))

            return True
        except Exception as e:
            print(e)
            print("Failed saving {}, continue anyway".format(name))
            return False

    def load_python_obj(self, path: str, name: str):
        """ Loads python object from disk if pickle """

        obj = None
        try:
            filepath = os.path.join(path, f'{name}.pickle')
            with (open(filepath, "rb")) as openfile:
                obj = pickle.load(openfile)

        except FileNotFoundError:
            print("{} not loaded because file is missing".format(name))
            return

        print("Loaded {}".format(name))
        return obj

    def python_obj_exists(self, path: str, name: str) -> bool:
        filepath = os.path.join(path, f'{name}.pickle')
        result = os.path.exists(filepath)
        return result

    def personal_deepcopy(self, obj):
        """ Deep copies any object faster than builtin """

        return pickle.loads(pickle.dumps(obj, protocol=-1))

    def duplicate_list(self, lst: list) -> list:
        """ shallow copies list """

        return [x for x in lst]

    def duplicate_set(self, st: set) -> set:
        """ shallow copies set """

        return {x for x in st}

    def duplicate_dict(self, dc) -> dict:
        """ shallow copies dictionary """
        return {key: dc[key] for key in dc}

    def duplicate_default_dict(self, dfdc, type_func, typ) -> defaultdict:
        """ shallow copies a defualt dictionary but gives the chance to also shallow copy its members """

        output = defaultdict(typ)
        for key in dfdc:
            output[key] = type_func(dfdc[key])
        return output

    def dump_only(self, obj):
        return pickle.dumps(obj, protocol=-1)

    def load_only(self, obj):
        return pickle.loads(obj)

    def save_figure(self, path: str, name: str, extension: str = 'png', no_axis: bool = True):
        if (no_axis):
            plt.axis('off')

        filepath = os.path.join(path, f'{name}.{extension}')
        plt.savefig(filepath, bbox_inches='tight')

    def set_date_stamp(self, addition=""):
        """ generates printable date stamp"""

        if (len(self.stamp) > 2):
            raise Exception(
                "Attempting to reset datestamp, but it was already set")

        self.actual_date = datetime.now()
        self.stamp = str(self.actual_date).split(".")[0].replace(
            " ", "_").replace(':', '.') + addition
        print(f"Made datestamp: {self.stamp}")
        return self.stamp
