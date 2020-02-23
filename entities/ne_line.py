class NELine:
    def __init__(self):
        self.tokens = []
        self.misc = []
        self.ne_coarse_lit = []
        self.ne_coarse_meto = []
        self.ne_fine_lit = []
        self.ne_fine_meto = []
        self.ne_fine_comp = []

        self.ne_nested = []
        self.nel_lit = []
        self.nel_meto = []

    def add_data(self, csv_row: dict):

        self._add_entity_if_available(csv_row, 'TOKEN', self.tokens)
        self._add_entity_if_available(csv_row, 'MISC', self.misc)
        self._add_entity_if_available(
            csv_row, 'NE-COARSE-LIT', self.ne_coarse_lit)
        self._add_entity_if_available(
            csv_row, 'NE-COARSE-METO', self.ne_coarse_meto)
        self._add_entity_if_available(csv_row, 'NE-FINE-LIT', self.ne_fine_lit)
        self._add_entity_if_available(
            csv_row, 'NE-FINE-METO', self.ne_fine_meto)
        self._add_entity_if_available(
            csv_row, 'NE-FINE-COMP', self.ne_fine_comp)
        self._add_entity_if_available(csv_row, 'NE-NESTED', self.ne_nested)
        self._add_entity_if_available(csv_row, 'NEL-LIT', self.nel_lit)
        self._add_entity_if_available(csv_row, 'NEL-METO', self.nel_meto)

    def get_text(self):
        text = ''
        for i, token in enumerate(self.tokens):
            text += token
            if 'NoSpaceAfter' not in self.misc[i] and 'EndOfLine' not in self.misc[i]:
                text += ' '

        return text

    def get_token_info(self, pos: int):
        return [
            self.tokens[pos],
            self.ne_coarse_lit[pos],
            self.ne_coarse_meto[pos],
            self.ne_fine_lit[pos],
            self.ne_fine_meto[pos],
            self.ne_fine_comp[pos],

            self.ne_nested[pos],
            self.nel_lit[pos],
            self.nel_meto[pos]
        ]

    def _add_entity_if_available(self, csv_row: dict, key: str, obj: list):
        if key in csv_row.keys():
            if csv_row[key] == 'O':
                obj.append(None)
            else:
                obj.append(csv_row[key])
