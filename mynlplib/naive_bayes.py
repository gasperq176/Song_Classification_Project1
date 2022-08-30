from mynlplib.constants import OFFSET
from mynlplib import clf_base, evaluation
from collections import Counter, defaultdict

import numpy as np

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    output_dict = Counter()
    for i in range(len(y)):
        if y[i] == label:
            output_dict += x[i]
     
    return output_dict

# deliverable 3.2
#Hint: note that this function takes the vocabulary as an argument. You have to assign a probability even for words that do not appear in documents with label  𝑦 , if they are in the vocabulary.
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    default_zeros = [0.0]*len(vocab)
    output_dict = dict(zip(list(vocab),default_zeros))
    corpus_counts = get_corpus_counts(x, y, label)
    
    corpus_words = list(corpus_counts.keys())
    corpus_countval = list(corpus_counts.values())
    total_count = sum(corpus_countval)
    
    for i in range(len(list(vocab))):
        word = list(vocab)[i]
        if word in corpus_words:
            output_dict[word] = np.log(corpus_counts[word] + smoothing) - np.log(len(vocab)*smoothing + total_count)
        else:
            output_dict[word] = np.log(smoothing)- np.log(len(vocab)*smoothing + total_count)
    
    #include OFFSET (I can see it is in the suggested solution, 
    #but adding this in will lead to the accumulated probability to 2 as exp(0) = 1
    #so I did not include it in the output)
   
    #output_dict[OFFSET] = 0.0
    
    return output_dict

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """

    labels = set(y)
    default_zeros = [0.0]*len(labels)
    priors = dict(zip(list(labels),default_zeros))
    aggregate_count = defaultdict(float)
    weights = defaultdict(float)
    
    #prior calculations
    for label in labels:
        label_count = Counter(y)[label]
        priors[label] = np.log(label_count) - np.log(len(y))
        
        #aggregate the features to each label
        agg_count = Counter()
        for i in range(len(y)):
            if y[i] == label:
                agg_count += x[i]
        aggregate_count[label] = agg_count    
        
    #total amount of vocab
    words_list = []
    for _ in x:
        words_list += list(_.keys())
    
    total_count_words = len(set(words_list))
    
    for label in labels:
        weights[(label, OFFSET)] = priors[label]
        est_xy_weights = estimate_pxy(x,y,label,smoothing,set(words_list))
        for word in list(est_xy_weights.keys()):
            weights[(label, word)] = est_xy_weights[word]

    return weights

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    '''
    scores = defaultdict(float)
    labels = list(set(y_tr))
    
    for smoother in smoothers:
        weights = estimate_nb(x_tr,y_tr,smoother)
        y_hat = clf_base.predict_all(x_dv,weights,labels)
        scores[smoother] = evaluation.acc(y_hat,y_dv)
       
    return clf_base.argmax(scores), scores
