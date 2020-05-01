from typing import List

from enums.entity_tag_type import EntityTagType

from entities.ne_line import NELine

class NECollection:
    def __init__(self):
        self.lines: List[NELine] = []

    def add_line(self, line: NELine):
        self.lines.append(line)

    def get_unique_entity_tags(self, entity_tag_type: EntityTagType):
        entities = []
        for line in self.lines:
            for coarse_entity in line.get_entity_tags(entity_tag_type):
                if coarse_entity not in entities:
                    entities.append(coarse_entity)

        return entities

    def __getitem__(self, idx) -> NELine:
        return self.lines[idx]

    def __len__(self):
        return len(self.lines)