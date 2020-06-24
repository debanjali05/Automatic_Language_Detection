# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model based on a combination of N-grams Language Model and Naive Bayes Classifier

Training of the Language detection model
"""

#import os
#import nltk
#import pickle

from data import load_dataset
#from utils import checkpoint_path
#from nltk.classify import NaiveBayesClassifier
from language_modelling import train_language_model

"""
def extract_features(document):
    
        extracting n-gram features 
    
    
    document_words = set(document)
    features = {}
    
    for ngram in word_features:
        features = dict(key = ngram, v= (ngram in document_words))
    
    return features #features
"""
print("\t-------Loading Training Set-------")
train_set = load_dataset("Training")  #generating the train set
print("\tLength of Training set:", len(train_set))

print("\t-------Start Training------")
#N-gram Language model based Naive Bayes Classifier for Language Detection 
for n in range(2,3):
    
    print(n,"-gram Language model based Naive Bayes Classifier for Language Detection")
    
    print("Generating",n,"-gram Language model------->")
    
    #Generating N-gram Language models
    
    train_language_model(n)
    """word_features = ngram.keys()
    training_set = nltk.classify.apply_features(extract_features, train_set)
    
    print("Training Naive Bayes Classifier-------")
    
    #Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(training_set)
    classifier_name = "classifier" + str(n)+".pickle"
    f = open(os.path.join(checkpoint_path,classifier_name), 'wb')
    pickle.dump(classifier, f)
    f.close()
    
    print("Model Saved at:",os.path.join(checkpoint_path,classifier_name))"""
    
print("\t-------End Training-------")