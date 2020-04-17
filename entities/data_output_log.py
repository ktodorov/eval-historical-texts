from __future__ import annotations


class DataOutputLog:
    def __init__(self):
        self.input_data = None
        self.output_data = None
        self.true_data = None

    def add_new_data(
            self,
            input_data: str = None,
            output_data: str = None,
            true_data: str = None):

        assert input_data is not None or output_data is not None or true_data is not None, 'At least one data must be populated'

        if input_data is not None:
            if self.input_data is None:
                self.input_data = []

            self.input_data.append(input_data)

        if output_data is not None:
            if self.output_data is None:
                self.output_data = []

            self.output_data.append(output_data)

        if true_data is not None:
            if self.true_data is None:
                self.true_data = []

            self.true_data.append(true_data)

    def extend(self, other_data_log: DataOutputLog):
        if other_data_log is None:
            return

        if other_data_log.input_data is not None:
            if self.input_data is None:
                self.input_data = []

            self.input_data.extend(other_data_log.input_data)

        if other_data_log.output_data is not None:
            if self.output_data is None:
                self.output_data = []

            self.output_data.extend(other_data_log.output_data)

        if other_data_log.true_data is not None:
            if self.true_data is None:
                self.true_data = []

            self.true_data.extend(other_data_log.true_data)

    def get_log_data(self) -> list:
        prepared_data = []
        columns = []

        if self.input_data is not None:
            columns.append('Input')
            prepared_data.append(self.input_data)

        if self.output_data is not None:
            columns.append('Output')
            prepared_data.append(self.output_data)

        if self.true_data is not None:
            columns.append('Target')
            prepared_data.append(self.true_data)

        data_length = len(self)
        result_data = [[x[i] for x in prepared_data]
                       for i in range(data_length)]

        return columns, result_data

    def __len__(self):
        available_lengths = [0]

        if self.input_data is not None:
            available_lengths.append(len(self.input_data))

        if self.output_data is not None:
            available_lengths.append(len(self.output_data))

        if self.true_data is not None:
            available_lengths.append(len(self.true_data))

        return max(available_lengths)
