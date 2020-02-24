from entities.ne_line import NELine

class NECollection:
    def __init__(self):
        self.lines = []

    def add_line(self, line: NELine):
        self.lines.append(line)

    def get_unique_coarse_entities(self):
        entities = []
        for line in self.lines:
            for coarse_entity in line.ne_coarse_lit:
                if coarse_entity not in entities:
                    entities.append(coarse_entity)

        return entities

    def __getitem__(self, idx) -> NELine:
        return self.lines[idx]

    def __len__(self):
        return len(self.lines)