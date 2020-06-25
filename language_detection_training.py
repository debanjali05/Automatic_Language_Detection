# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model based on N-grams

Training of the Language detection model
"""

#import os
#import nltk
#import pickle

from data import load_dataset
#from utils import checkpoint_path
#from nltk.classify import NaiveBayesClassifier
from language_modelling import train_language_model

print("\t-------Loading Training Set-------")
train_set = load_dataset("Training")  #generating the train set
print("Length of Training set:", len(train_set))

print("\t-------Start Training------")
#N-gram Language model based Naive Bayes Classifier for Language Detection 
for n in range(2,3):
    
    print("Generating",n,"-gram Language model for Language Detection------->")
    
    #Generating N-gram Language models
    
    train_language_model(n)
    
print("\t-------End Training-------")