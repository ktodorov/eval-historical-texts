import os
import csv

from entities.ne_collection import NECollection
from entities.ne_line import NELine


def preprocess_data(file_path: str) -> NECollection:
    if not os.path.exists(file_path):
        raise Exception('NER File not found')

    collection = NECollection()

    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, dialect='excel-tab')
        current_sentence = NELine()

        for i, row in enumerate(reader):
            if row['TOKEN'].startswith('#'):
                continue

            current_sentence.add_data(row)

            if 'EndOfLine' in row['MISC']:
                collection.add_line(current_sentence)
                current_sentence = NELine()

    return collection