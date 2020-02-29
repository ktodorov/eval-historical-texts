from services.tokenizer_service import TokenizerService


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
        self._add_entity_if_available(
            csv_row, 'MISC', self.misc, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-COARSE-LIT', self.ne_coarse_lit, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-COARSE-METO', self.ne_coarse_meto, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-FINE-LIT', self.ne_fine_lit, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-FINE-METO', self.ne_fine_meto, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-FINE-COMP', self.ne_fine_comp, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NE-NESTED', self.ne_nested, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NEL-LIT', self.nel_lit, use_none_if_empty=True)
        self._add_entity_if_available(
            csv_row, 'NEL-METO', self.nel_meto, use_none_if_empty=True)

    def tokenize_text(self, tokenizer_service: TokenizerService):

        if (self.tokens[0] == '«'):
            a = 0

        text = self.get_text()
        offsets = self.get_token_offsets()


        token_ids, encoded_tokens, encoded_offsets, _ = tokenizer_service.encode_sequence(
            text)

        # it means that the tokenizer has split some of the words, therefore we need to add
        # those tokens to our collection and repeat the entity labels for the new sub-tokens
        if len(encoded_tokens) > len(self.tokens):
            new_misc = self.misc
            new_ne_coarse_lit = self.ne_coarse_lit
            new_ne_coarse_meto = self.ne_coarse_meto
            new_ne_fine_lit = self.ne_fine_lit
            new_ne_fine_meto = self.ne_fine_meto
            new_ne_fine_comp = self.ne_fine_comp

            new_ne_nested = self.ne_nested
            new_nel_lit = self.nel_lit
            new_nel_meto = self.nel_meto

            corresponding_counter = 0

            for i, token in enumerate(self.tokens):
                while corresponding_counter < len(encoded_tokens) and encoded_offsets[corresponding_counter][1] < offsets[i][1]:

                    # we copy the value of the original token
                    new_misc.insert(corresponding_counter, self.misc[i])
                    new_ne_coarse_lit.insert(
                        corresponding_counter, self.ne_coarse_lit[i])
                    new_ne_coarse_meto.insert(
                        corresponding_counter, self.ne_coarse_meto[i])
                    new_ne_fine_lit.insert(
                        corresponding_counter, self.ne_fine_lit[i])
                    new_ne_fine_meto.insert(
                        corresponding_counter, self.ne_fine_meto[i])
                    new_ne_fine_comp.insert(
                        corresponding_counter, self.ne_fine_comp[i])
                    new_ne_nested.insert(
                        corresponding_counter, self.ne_nested[i])
                    new_nel_lit.insert(corresponding_counter, self.nel_lit[i])
                    new_nel_meto.insert(
                        corresponding_counter, self.nel_meto[i])

                    corresponding_counter += 1

                corresponding_counter += 1

            self.misc = new_misc
            self.ne_coarse_lit = new_ne_coarse_lit
            self.ne_coarse_meto = new_ne_coarse_meto
            self.ne_fine_lit = new_ne_fine_lit
            self.ne_fine_meto = new_ne_fine_meto
            self.ne_fine_comp = new_ne_fine_comp

            self.ne_nested = new_ne_nested
            self.nel_lit = new_nel_lit
            self.nel_meto = new_nel_meto

        assert len(token_ids) == len(encoded_tokens)
        assert len(token_ids) == len(self.ne_coarse_lit)
        self.tokens = encoded_tokens

        self.token_ids = token_ids

    def get_text(self):
        text = ''

        for i, token in enumerate(self.tokens):
            if token.startswith('##'):
                text = text[:-1]
                text += token[2:]
            else:
                text += token

            # if 'NoSpaceAfter' not in self.misc[i] and 'EndOfLine' not in self.misc[i]:
            text += ' '

        return text

    def get_token_offsets(self):
        offsets = []

        last_end = 0

        for i, token in enumerate(self.tokens):
            current_start = last_end
            current_end = current_start + len(token)
            last_end += len(token)

            # if 'NoSpaceAfter' not in self.misc[i] and 'EndOfLine' not in self.misc[i]:
            last_end += 1

            offsets.append((current_start, current_end))

        return offsets

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

    def _add_entity_if_available(self, csv_row: dict, key: str, obj: list, use_none_if_empty: bool = False):
        if key in csv_row.keys():
            if csv_row[key] == 'O' and use_none_if_empty:
                obj.append(None)
            else:
                obj.append(csv_row[key])
