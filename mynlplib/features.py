from mynlplib.constants import OFFSET
from operator import itemgetter
import numpy as np
import copy as cp
import pandas as pd
import torch

# deliverable 6.1
def get_top_features_for_label(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    weights_fixedLabel = []
    weights_items = list(weights.items())
    weights_keys = list(weights.keys())
    weights_keys_labels = list(zip(*weights_keys))[0]
    
    #find all keys from the weights dict only for the given label
    for i in range(len(weights_keys)):
        if weights_keys_labels[i] == label:
            weights_fixedLabel.append(weights_items[i])
        
    return sorted(weights_fixedLabel, key = itemgetter(1), reverse = True)[:k]

# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    output = []
    vocab = sorted(vocab)
    label_index = label_set.index(label)
    
    #retrieve weights from the model
    weights_at_label = model.Linear.weight[label_index] 
    
    #indeces of sorted weights
    indices = sorted(range(len(weights_at_label)), key=lambda i: weights_at_label[i], reverse=True)[:k]
    
    for i in indices:
        output.append(vocab[i])
    
    return output

# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    tokens = sum(counts)
    types = list(filter(lambda x: (x > 0), counts))
    diff_types = len(types)
    
    #total tokens / different types
    return tokens/diff_types

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    N = data.shape[0]
    V = data.shape[1]
    output = np.zeros(shape=(N,V+7))
    #calculate token type ratio
    process = [get_token_type_ratio(data[i]) for i in range(len(data))]
    #assign the results into bins
    bins = [0,1,2,3,4,5,6,float("inf")]
    bin_indices = np.digitize(process, bins, right=False)
    
    #Use np.concatenate or np.hstack to concatenate your result to the variable X_tr.
    for i in range(N):
        index_bin = bin_indices[i]
        bin_binaries = [0]*7
        bin_binaries[index_bin-2] = 1
        
        arr_1 = np.array(data[i])
        arr_2 = np.array(bin_binaries)
        comb = np.hstack((arr_1,arr_2))
        
        output[i] = comb
        
    return output
