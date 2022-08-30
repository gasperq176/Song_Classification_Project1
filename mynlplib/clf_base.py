from collections import defaultdict
from mynlplib.constants import OFFSET
import numpy as np
import collections

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features,label):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    dic = defaultdict(float)
    
    for key, value in base_features.items():
        dic[(label, key)] = float(value)
    
    dic[(label, OFFSET)] = 1.0
    return  dic

# deliverable 2.2
def predict(base_features,weights,labels):
    '''
    prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''    
    default_zeros = [0.0]*len(labels)
    scores = dict(zip(list(labels),default_zeros))
    features_keys = list(base_features.keys())
    
    if weights:
        weights_keys = list(weights.keys())
        weights_val = list(weights.values())
        weights_keys_labels = list(zip(*weights_keys))[0]
        weights_keys_features = list(zip(*weights_keys))[1]
        
        for i in range(len(weights_keys_labels)):
            label = weights_keys_labels[i]
            
            if weights_keys_features[i] in features_keys:
                word_key = weights_keys_features[i]
                scores[label] += weights_val[i] * base_features[word_key]
                
            if weights_keys_features[i] == OFFSET:
                scores[label] += weights_val[i]
    
    return argmax(scores), scores

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat