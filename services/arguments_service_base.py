import argparse

from typing import Dict


class ArgumentsServiceBase:
    def __init__(self):
        self._arguments: argparse.Namespace = {}
        self._parse_arguments()

    def get_argument(self, key: str) -> object:
        """Returns an argument value from the list of registered arguments

        :param key: key of the argument
        :type key: str
        :raises LookupError: if no argument is found, lookup error will be raised
        :return: the argument value
        :rtype: object
        """
        if key not in self._arguments.keys():
            raise LookupError(f'{key} not found in arguments')

        return self._arguments[key]

    def print_arguments(self):
        """Prints the arguments which the program was initialized with
        """
        print(f'Arguments initialized: {self._arguments}')

    def _parse_arguments(self):
        parser = argparse.ArgumentParser()

        self._add_specific_arguments(parser)
        self._arguments: Dict[str, object] = vars(parser.parse_args())

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        pass
