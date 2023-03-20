# Imports
import numpy as np
from typing import List, Tuple
from numpy._typing import ArrayLike
from numpy import random
from datetime import datetime

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    sampled_seqs = []
    sampled_labels = []
    for idx, i in enumerate(range(len(seqs))):
        random.seed()
        rand = random.randint(0,len(seqs))
        sampled_seqs.append(seqs[rand])
        sampled_labels.append(labels[rand])
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0],
           'G': [0, 0, 0, 1], 'Other': [0, 0, 0, 0]}
    encoded_lst = [] = []
    for s in seq_arr:
        for base in s:
            if base in map.keys():
                for i in map[base]:
                    encoded_lst.append(i)
            else:
                for i in map['Other']:
                    encoded_lst.append(i)
    return encoded_lst

seq = 'ATCGF'
x = one_hot_encode_seqs(seq)
print(x)