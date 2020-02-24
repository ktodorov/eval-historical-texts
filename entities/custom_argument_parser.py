import argparse


class CustomArgumentParser(argparse.ArgumentParser):
    def __init__(self, raise_errors_on_invalid_args: bool = True):
        super().__init__()

        self._raise_errors_on_invalid_args = raise_errors_on_invalid_args

    def error(self, message):
        if self._raise_errors_on_invalid_args:
            super().error(message)