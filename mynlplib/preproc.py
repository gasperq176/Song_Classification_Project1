from collections import Counter

import pandas as pd
import numpy as np
import copy as cp

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    word_list = text.split()
    return Counter(word_list)

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counts = Counter()
    # YOUR CODE GOES HERE
    for bow in bags_of_words:
        counts = counts + bow
    
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    bow1_s = set(bow1.keys())
    bow2_s = set(bow2.keys())
    
    return bow1_s.difference(bow2_s)

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    pruned_train = {word: count for word, count in training_counts.items() if count >= min_counts}
    delete = []
    vocab = set(pruned_train.keys())
    return_target_data = cp.deepcopy(target_data)
    
    for bow in return_target_data:
        for w in list(bow.keys()):
            if w not in vocab:
                del bow[w]

    return return_target_data, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    #one approach is to start with numpy.zeros((height,width)), and then fill in the cells by iterating through the bag-of-words list
    vocab = sorted(vocab)
    height = len(bags_of_words)
    width = len(vocab)
    output = np.zeros((height,width))
                                            
    for h in range(height):
        bow = bags_of_words[h]
        for w in range(width):
            if vocab[w] in list(bow.keys()):
                output[h][w] = bow[vocab[w]]
                         
    return output


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
