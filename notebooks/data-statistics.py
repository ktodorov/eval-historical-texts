#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from tokenizers import BertWordPieceTokenizer

import jellyfish

plt.rcParams['figure.dpi'] = 120


# In[2]:


full_ocr_path = os.path.join('results', 'combined_ocr.pickle')
full_gs_path = os.path.join('results', 'combined_gs.pickle')
data_path = 'results'

def read_ocr_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as language_file:
        text_data: List[str] = language_file.read().split('\n')

        return(text_data[1][start_position:], text_data[2][start_position:])

def save_data_files():
    start_position = 14

    ocr_aligned_lengths = []
    gs_aligned_lengths = []
    file_paths = []

    for i, file_name in enumerate(os.listdir(data_path)):
        file_paths.append(os.path.join(data_path, file_name))

    number_of_files = len(file_paths)
    file_data = []
    for i, file_path in enumerate(file_paths):
        print(f'{i}/{number_of_files}             \r', end='')
        file_data.append(read_ocr_file(file_path))
        
    ocr_file_data = [x[0] for x in file_data]
    gs_file_data = [x[1] for x in file_data]
    
    with open(full_ocr_path, 'wb') as ocr_handle:
        pickle.dump(ocr_file_data, ocr_handle, protocol=-1)
    
    with open(full_gs_path, 'wb') as gs_handle:
        pickle.dump(gs_file_data, gs_handle, protocol=-1)
        
    return ocr_file_data, gs_file_data

if not os.path.exists(full_ocr_path) or not os.path.exists(full_gs_path):
    ocr_file_data, gs_file_data = save_data_files()
else:
    with open(full_ocr_path, 'rb') as ocr_handle:
        ocr_file_data = pickle.load(ocr_handle)
    
    with open(full_gs_path, 'rb') as gs_handle:
        gs_file_data = pickle.load(gs_handle)


# In[3]:


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_list_histogram(lst, title: str):
    labels, values = zip(*Counter(lst).items())
    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.title(title)

    plt.show()
    
def print_statistics(lst, title:str):
    max_value = np.max(lst)
    min_value = np.min(lst)
    avg_value = np.mean(lst)
    
    print(f'{title}:\nMAX: {max_value}\nMIN: {min_value}\nAVG: {avg_value}')


# In[4]:


ocr_lengths = np.array([len(x) for x in ocr_file_data])
gs_lengths = np.array([len(x) for x in gs_file_data])

# indices = np.argsort(ocr_lengths)
# print(indices)


# print(ocr_file_data[0])
# ocr_file_data_np = np.array(ocr_file_data, dtype=object)
# gs_file_data_np = np.array(gs_file_data)[indices]


# plot_list_histogram(ocr_lengths, 'OCR')
# print_statistics(ocr_lengths, 'OCR')

# plot_list_histogram(gs_lengths, 'GS')
# print_statistics(gs_lengths, 'GS')


# In[5]:


full_ocr_tokens_path = os.path.join('results', 'combined_ocr_tokens.pickle')
full_gs_tokens_path = os.path.join('results', 'combined_gs_tokens.pickle')

vocab_path = os.path.join('vocabularies', 'bert-base-cased-vocab.txt')
tokenizer = BertWordPieceTokenizer(vocab_path)

if not os.path.exists(full_ocr_tokens_path) or not os.path.exists(full_gs_tokens_path):        
    ocr_tokens = []
    gs_tokens = []
    for i in range(len(ocr_file_data)):
        current_ids = tokenizer.encode(ocr_file_data[i]).ids
        if len(current_ids) > 2000:
            continue
            
        ocr_tokens.append(current_ids)
        gs_tokens.append(tokenizer.encode(gs_file_data[i]).ids)
    
    with open(full_ocr_tokens_path, 'wb') as ocr_handle:
        pickle.dump(ocr_tokens, ocr_handle, protocol=-1)
    
    with open(full_gs_tokens_path, 'wb') as gs_handle:
        pickle.dump(gs_tokens, gs_handle, protocol=-1)
else:
    with open(full_ocr_tokens_path, 'rb') as ocr_handle:
        ocr_tokens = pickle.load(ocr_handle)
    
    with open(full_gs_tokens_path, 'rb') as gs_handle:
        gs_tokens = pickle.load(gs_handle)


# In[6]:


ocr_tokens_lengths = np.array([len(x) for x in ocr_tokens])
gs_tokens_lengths = np.array([len(x) for x in gs_tokens])

# indices = np.argsort(ocr_tokens_lengths)

# ocr_tokens_lengths = np.array(ocr_tokens_lengths)[indices]
# gs_tokens_lengths = np.array(gs_tokens_lengths)[indices]
# ocr_tokens = np.array(ocr_tokens)[indices]
# gs_tokens = np.array(gs_tokens)[indices]

# print(f'OCR - Less than 2000 length: {len(ocr_tokens_lengths[ocr_tokens_lengths <= 2000]) / len(ocr_tokens_lengths) * 100}')
# print(f'GS  - Less than 2000 length: {len(gs_tokens_lengths[gs_tokens_lengths <= 2000]) / len(gs_tokens_lengths) * 100}')

# plot_list_histogram(ocr_tokens_lengths, 'OCR - Tokens')
# print_statistics(ocr_tokens_lengths, 'OCR - Tokens')

# plot_list_histogram(gs_tokens_lengths, 'GS - Tokens')
# print_statistics(gs_tokens_lengths, 'GS - Tokens')


# In[7]:


def calculate_jaccard_similarity(list1: list, list2: list) -> float:
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

def calculate_levenshtein_distance(string1: str, string2: str) -> int:
    result = jellyfish.levenshtein_distance(string1, string2)
    return result


# In[9]:


def save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances):
    metrics_path = os.path.join('results', 'metrics.pickle')

    metrics_obj = {
        'token_pairs': token_pairs,
        'decoded_pairs': decoded_pairs,
        'jaccard_similarities': jaccard_similarities,
        'levenshtein_distances': levenshtein_distances,
    }

    with open(metrics_path, 'wb') as metrics_handle:
        pickle.dump(metrics_obj, metrics_handle, protocol=-1)
        
    print('Saved metrics')

def load_metrics_obj():
    metrics_path = os.path.join('results', 'metrics.pickle')
    if not os.path.exists(metrics_path):
        return (None, None, None, None)

    with open(metrics_path, 'rb') as metrics_handle:
        metrics_obj = pickle.load(metrics_handle)
        
    return (metrics_obj['token_pairs'],
            metrics_obj['decoded_pairs'],
            metrics_obj['jaccard_similarities'],
            metrics_obj['levenshtein_distances'])
        
token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances = load_metrics_obj()

if not token_pairs:
    token_pairs = [([tokenizer.id_to_token(x) for x in ocr_tokens[i]], [tokenizer.id_to_token(x) for x in gs_tokens[i]]) for i in range(len(ocr_tokens))]
    save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances)
    
if not decoded_pairs:
    decoded_pairs = [(tokenizer.decode(ocr_tokens[i]), tokenizer.decode(gs_tokens[i])) for i in range(len(ocr_tokens))]
    save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances)
    
all_pairs = len(token_pairs)
if not jaccard_similarities:
    jaccard_similarities = []
    for i, token_pair in enumerate(token_pairs):
        jaccard_similarities.append(calculate_jaccard_similarity(token_pair[0], token_pair[1]))
    
    save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances)
    
if not levenshtein_distances:
    levenshtein_distances = []
    
if len(levenshtein_distances) < all_pairs:
    for i, decoded_pair in enumerate(decoded_pairs):
        if i < len(levenshtein_distances):
            continue
            
        print(f'LEVENSHTEIN - {i}/{all_pairs}             \r', end='')
        levenshtein_distances.append(calculate_levenshtein_distance(decoded_pair[0], decoded_pair[1]))
        
        if i % 5000 == 0:
            save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances)
    
    save_metrics_obj(token_pairs, decoded_pairs, jaccard_similarities, levenshtein_distances)


# In[ ]:


metrics_path = os.path.join('results', 'combined_ocr_tokens.pickle')

metrics_obj = {
    'token_pairs': token_pairs,
    'decoded_pairs': decoded_pairs,
    'jaccard_similarities': jaccard_similarities,
    'levenshtein_distances': levenshtein_distances,
}

with open(metrics_path, 'wb') as metrics_handle:
    pickle.dump(metrics_obj, metrics_handle, protocol=-1)

