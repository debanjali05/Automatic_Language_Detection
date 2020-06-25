# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Implementation of a Language detection model based on N-grams

Training of the Language detection model
"""

from utils import n
from data import load_dataset
from language_modelling import train_language_model

#reading the training data text files in unicode
print("\t-------Loading Training Set-------")
train_set = load_dataset("Training")  #generating the train set
print("Length of Training set:", len(train_set))

print("\t-------Start Training------")

#N-gram Language model based Naive Bayes Classifier for Language Detection 
print("Generating",n,"-gram Language model for Language Detection------->")
    
#Generating N-gram Language models   
train_language_model(n)
    
print("\t-------End Training-------")