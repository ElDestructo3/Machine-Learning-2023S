import numpy # linear algebra
import csv # data processing, CSV file I/O
import pandas # data processing, CSV file I/O
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import naive_bayes # sklearn Naive Bayes
import operator
from math import *
from collections import Counter


def separate_by_class(dataset):
    separated_by_class = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        if (sample[-1] not in separated_by_class):
            separated_by_class[sample[-1]] = []
        separated_by_class[sample[-1]].append(sample)
    return separated_by_class

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return sqrt(variance)

def continuous_class_probabilities(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def create_freq_distribution(dataset):
    dict = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        prediction = sample[-1]
        sample = sample[:-2]
        for j in range(len(sample)):
            if prediction not in dict:
                dict[prediction] = {}
            if j not in dict[prediction]:
                dict[prediction][j] = {}
            if sample[j] not in dict[prediction][j]:
                dict[prediction][j][sample[j]] = 0
            dict[prediction][j][sample[j]] += 1
    
    return dict

def get_class_probabilities(freq_distribution, index, class_val, prediction):
    num_samples_with_class = freq_distribution[prediction][index][class_val]
    num_samples = sum(freq_distribution[prediction][index].values())
    print(num_samples_with_class)
    print(num_samples)
    print(num_samples_with_class / num_samples)
    return num_samples_with_class / num_samples

dataset = pandas.read_csv('hospital.csv')
dataset = dataset.values.tolist()
new_dataset_freq = create_freq_distribution(dataset)
#print(new_dataset_freq)
get_class_probabilities(new_dataset_freq, 2, 'c', '20-30')