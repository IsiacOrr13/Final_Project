# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from numpy import random

def sample_seqs(seqs_pos: List[str],seqs_neg: List[str],
                labels_pos: List[bool], labels_neg: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs_pos: List[str]
            List of all positive sequences.
        seqs_neg: List[str]
            List of all negative sequences.
        labels_pos: List[bool]
            List of positive labels
        labels_neg: List[bool]
            List of negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Rand decides if the next seq in the sample is positive or negative for TF
    # Rand2 picks a random positive or negative seq (depending on Rand) to add to the sample

    sampled_seqs = []
    sampled_labels = []
    seq_len_total = 10000
    for i in range(seq_len_total):
        random.seed()
        rand = random.randint(0,2)
        if rand == 1:
            random.seed()
            rand_2 = random.randint(0, len(seqs_pos))
            sampled_seqs.append(seqs_pos[rand_2])
            sampled_labels.append([labels_pos[rand_2]])
        if rand == 0:
            random.seed()
            rand_2 = random.randint(0, len(seqs_neg))
            sampled_seqs.append(seqs_neg[rand_2])
            sampled_labels.append([labels_neg[rand_2]])
    return np.array(sampled_seqs), np.array(sampled_labels)

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
    encoded_lst = []
    for s in seq_arr:
        encode = []
        for base in s:
            if base in map.keys():
                for i in map[base]:
                    encode.append(i)
            else:
                for i in map['Other']:
                    encode.append(i)
        encoded_lst.append(encode)
    return np.array(encoded_lst)


