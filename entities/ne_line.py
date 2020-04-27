from typing import Dict, List
from copy import deepcopy
import re

from services.tokenize.base_tokenize_service import BaseTokenizeService


class NELine:
    def __init__(self):
        self.tokens: List[str] = []
        self.misc = []
        self.ne_coarse_lit = []
        self.ne_coarse_meto = []
        self.ne_fine_lit = []
        self.ne_fine_meto = []
        self.ne_fine_comp = []

        self.ne_nested = []
        self.nel_lit = []
        self.nel_meto = []
        self.original_length = 0
        self.position_changes: Dict[int, List[int]] = None

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

    def _insert_entity_tag(self, list_to_modify: list, position: int, tag: str):
        tag_to_insert = tag
        if tag.startswith('B-'):
            tag_to_insert = f'I-{tag[2:]}'

        list_to_modify.insert(position, tag_to_insert)

    def tokenize_text(
        self,
        tokenize_service: BaseTokenizeService,
        replace_all_numbers: bool = False):
        if replace_all_numbers:
            self.tokens = [re.sub('(([0-9]+)|(([0-9]*)\.([0-9]*)))', '0', token) for token in self.tokens] # replace digit with 0

        self.original_length = len(self.tokens)
        text = self.get_text()
        offsets = self.get_token_offsets()

        token_ids, encoded_tokens, encoded_offsets, _ = tokenize_service.encode_sequence(
            text)

        # it means that the tokenizer has split some of the words, therefore we need to add
        # those tokens to our collection and repeat the entity labels for the new sub-tokens
        position_changes = {i: [i] for i in range(len(self.tokens))}
        if len(encoded_tokens) > len(self.tokens):
            new_misc = deepcopy(self.misc)
            new_ne_coarse_lit = deepcopy(self.ne_coarse_lit)
            new_ne_coarse_meto = deepcopy(self.ne_coarse_meto)
            new_ne_fine_lit = deepcopy(self.ne_fine_lit)
            new_ne_fine_meto = deepcopy(self.ne_fine_meto)
            new_ne_fine_comp = deepcopy(self.ne_fine_comp)

            new_ne_nested = deepcopy(self.ne_nested)
            new_nel_lit = deepcopy(self.nel_lit)
            new_nel_meto = deepcopy(self.nel_meto)

            position_changes = {}
            corresponding_counter = 0

            for i, token in enumerate(self.tokens):
                position_changes[i] = [corresponding_counter]

                while corresponding_counter < len(encoded_tokens) and encoded_offsets[corresponding_counter][1] < offsets[i][1]:

                    # we copy the value of the original token
                    self._insert_entity_tag(
                        new_misc, corresponding_counter+1, self.misc[i])
                    self._insert_entity_tag(
                        new_ne_coarse_lit, corresponding_counter+1, self.ne_coarse_lit[i])
                    self._insert_entity_tag(
                        new_ne_coarse_meto, corresponding_counter+1, self.ne_coarse_meto[i])
                    self._insert_entity_tag(
                        new_ne_fine_lit, corresponding_counter+1, self.ne_fine_lit[i])
                    self._insert_entity_tag(
                        new_ne_fine_meto, corresponding_counter+1, self.ne_fine_meto[i])
                    self._insert_entity_tag(
                        new_ne_fine_comp, corresponding_counter+1, self.ne_fine_comp[i])
                    self._insert_entity_tag(
                        new_ne_nested, corresponding_counter+1, self.ne_nested[i])
                    self._insert_entity_tag(
                        new_nel_lit, corresponding_counter+1, self.nel_lit[i])
                    self._insert_entity_tag(
                        new_nel_meto, corresponding_counter+1, self.nel_meto[i])

                    corresponding_counter += 1
                    position_changes[i].append(corresponding_counter)

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

        self.position_changes = position_changes

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
            if csv_row[key] == '' and use_none_if_empty:
                obj.append(None)
            else:
                obj.append(csv_row[key])
