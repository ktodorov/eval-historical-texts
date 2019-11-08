import argparse


class ArgumentsServiceBase:
    def __init__(self):
        self._arguments: argparse.Namespace = {}

    def parse_arguments(self):
        parser = argparse.ArgumentParser()

        self._add_specific_arguments(parser)

        self._arguments = vars(parser.parse_args())

        self.print_arguments()

    def get_argument(self, key: str):
        return self._arguments[key]

    def print_arguments(self):
        print(f'Arguments initialized: {self._arguments}')

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        pass
