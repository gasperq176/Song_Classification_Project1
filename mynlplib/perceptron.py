from collections import defaultdict
from mynlplib.clf_base import predict,make_feature_vector
from mynlplib.naive_bayes import clf_base, evaluation
import copy as cp

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    #You should find the update by f(x,y) - f(x, y_hat). And then add the values in the update dict to the weights. And you should only return update, not the weights
    
    #initiate update output
    update = defaultdict(float)
    
    #get y_hat
    y_hat = predict(x, weights, labels)[0]
    
    
    f_xy = make_feature_vector(x, y)
    f_xy_hat = make_feature_vector(x, y_hat)
    
    f_xy_labels = list(zip(*f_xy.keys()))[0]
    f_xy_features = list(zip(*f_xy.keys()))[1]
    f_xy_values = list(f_xy.values())
  
    f_xy_hat_labels = list(zip(*f_xy_hat.keys()))[0]
    f_xy_hat_features = list(zip(*f_xy_hat.keys()))[1]
    f_xy_hat_values = list(f_xy_hat.values())
    
    for i in range(len(f_xy.keys())):   
        if f_xy_labels[i] != f_xy_hat_labels[i] and f_xy_features[i] == f_xy_hat_features[i]:
            update[(f_xy_labels[i], f_xy_features[i])] = f_xy_values[i] 
            update[(f_xy_hat_labels[i], f_xy_hat_features[i])] = -f_xy_hat_values[i]
    
    return update

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            update = perceptron_update(x_i,y_i,weights,labels)
            for key in update.keys():
                weights[key] += update[key]
            
        weight_history.append(weights.copy())
    return weights, weight_history
