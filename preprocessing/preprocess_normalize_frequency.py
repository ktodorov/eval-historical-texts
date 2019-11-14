from gensim.models.word2vec import PathLineSentences
from collections import defaultdict
import os
import sys

sys.path.append('..')
from utils import path_utils


def generate_frequency_matrices(is_norm : bool = True, is_absolute: bool = True, threshold: int = 0.0003):
    data_folder = path_utils.combine_path('..', 'data', 'semeval_trial_data')
    english_data_folder = path_utils.combine_path(
        data_folder, 'corpora', 'english')

    corpus1_path = path_utils.combine_path(
        english_data_folder, 'corpus1')
    corpus2_path = path_utils.combine_path(
        english_data_folder, 'corpus2')

    # Load targets
    target_path = path_utils.combine_path(data_folder, 'targets', 'english.txt')

    with open(target_path, 'r', encoding='utf-8') as f_in:
        targets = [line.strip() for line in f_in]

    string2value1 = generate_frequency_matrices_for_corpus(
        data_folder, corpus1_path, 'freq_corpus1.csv', targets, is_norm)
    string2value2 = generate_frequency_matrices_for_corpus(
        data_folder, corpus2_path, 'freq_corpus2.csv', targets, is_norm)

    results_path = path_utils.combine_path(data_folder, 'results', create_if_missing=True)
    difference_file = os.path.join(results_path, 'fd.csv')
    with open(difference_file, 'w', encoding='utf-8') as f_out:
        for string in targets:
            try:
                if is_absolute:
                    f_out.write('\t'.join((string, str(abs(string2value2[string]-string2value1[string]))+'\n')))
                else:
                    f_out.write('\t'.join((string, str(string2value2[string]-string2value1[string])+'\n')))
            except KeyError:
                f_out.write(f_out, '\t'.join((string, 'nan'+'\n')))

    with open(difference_file, 'r', encoding='utf-8') as f_in:
        string_value = [( line.strip().split('\t')[0], float(line.strip().split('\t')[1]) ) for line in f_in]

    # Get strings and string-value map
    strings = [s for (s,v) in string_value]
    string2value = dict(string_value)

    for string in strings:
        if string2value[string]<threshold:
            print(f'{string}-0')
        elif string2value[string]>=threshold:
            print(f'{string}-1')
        else:
            print(f'{string}-nan')

def generate_frequency_matrices_for_corpus(data_path: str, corpus_path: str, output_file_name: str, targets, is_norm=True):
    # Get sentence iterator
    sentences = PathLineSentences(corpus_path)

    # Initialize frequency dictionary
    freqs = defaultdict(int)

    # Iterate over sentences and words
    corpusSize = 0
    for sentence in sentences:
        for word in sentence:
            corpusSize += 1
            freqs[word] = freqs[word] + 1

    frequencies_path = path_utils.combine_path(
        data_path, 'frequencies', create_if_missing=True)
    english_frequencies_path = path_utils.combine_path(
        frequencies_path, 'english', create_if_missing=True)
    frequencies_file_path = os.path.join(
        english_frequencies_path, output_file_name)

    with open(frequencies_file_path, 'w', encoding='utf-8') as f_out:
        for word in targets:
            if word in freqs:
                if is_norm:
                    # Normalize by total corpus frequency
                    freqs[word] = float(freqs[word])/corpusSize
                f_out.write('\t'.join((word, str(freqs[word])+'\n')))
            else:
                f_out.write('\t'.join((word, 'nan'+'\n')))

    with open(frequencies_file_path, 'r', encoding='utf-8') as f_in:
        result = dict([( line.strip().split('\t')[0], float(line.strip().split('\t')[1]) ) for line in f_in])

    return result


generate_frequency_matrices()
