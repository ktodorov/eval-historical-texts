import argparse

from typing import Dict

class ArgumentsServiceBase:
    def __init__(self):
        self._arguments: argparse.Namespace = {}

        self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        self._add_specific_arguments(parser)
        self._arguments : Dict[str, object] = vars(parser.parse_args())
        self.print_arguments()

    def get_argument(self, key: str) -> object:
        if key not in self._arguments.keys():
            raise Exception(f'{key} not found in arguments')

        return self._arguments[key]

    def print_arguments(self):
        print(f'Arguments initialized: {self._arguments}')

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        pass
