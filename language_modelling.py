# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Task: Implementation of a Language detection model based on N-grams

Generating N-gram Language models
"""

import os
import math
import numpy as np

from data import build_dataset
from utils import train_path, train_lang_path, checkpoint_path, lang
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder

def calculate_total_ngrams(model):
    """
       calculating total number of N-grams
       
       model: list of saved models
    """
    no_of_ngrams = []
    for i,l in enumerate(lang):
        #model = np.load(lang+".npy")
        total = 0
        for key,v in model[i]:
            total = total + v
        no_of_ngrams.append(total)
    return no_of_ngrams

def generating_ngrams(words, n):
    """
       generating individual language models
    
        words: list of words present in a particular language (list(string))
        n: value of n for generating n-grams, values are 2,3,4 (int)
    """
    #Generating N-grams
    if n == 2:
        finder = BigramCollocationFinder.from_words(words) #2-grams
    elif n == 3:
        finder = TrigramCollocationFinder.from_words(words) #3-grams
    elif n == 4:
        finder = QuadgramCollocationFinder.from_words(words) #4-grams
    else:
        print("Incorrect value of n")
    
    return finder #ngrams


def train_language_model(n):
    """
        generating n-gram language models for n=2,3,4
    
        n: value of n for generating n-grams, values are 2,3,4 (int)
    """
    
    #Generating N-gram language model
    for i,l in enumerate(lang):
        
        path = os.path.join(train_path,train_lang_path[i]) #path to train text
        _, word_list = build_dataset(path,l) #generating word list individual language
       
        ngram = generating_ngrams(word_list, n) #generating N-gram
        ngram.apply_freq_filter(5) #filtering N-grams with frequency less that 5
        ngram_model = ngram.ngram_fd.items() 
        ngram_model = sorted(ngram.ngram_fd.items(), key=lambda item: item[1],reverse=True)  
 
        print("Length of",l,"model:",len(ngram_model))
        print(ngram_model)
        
        np.save(os.path.join(checkpoint_path,l+str(n)+"gram.npy"),ngram_model)

def predict_language_model(words,n,model):
    """
       predicting language
    
        words: list of words present in a particular language (list(string))
        n: value of n for generating n-grams, values are 2,3,4 (int)
        model: list of saved models
    """
    
    ngram = generating_ngrams(words,n) #generating N-gram
    total = calculate_total_ngrams(model) #calculating total N-grams
 
    freq_sum = np.zeros(5)
    
    #Calculating probabilities
    for k,v in ngram.ngram_fd.items():
        there = 0
        for i,l in enumerate(lang):
            for key,f in model[i]:
                if k == key:
                    freq_sum[i] = freq_sum[i]+ math.log(f/total[i],2)*(-1) #log(p1*p2) = log(p1) +log(p2)
                    there = 1
                    break
        if not there:
             freq_sum[i] = freq_sum[i]+0
 
    #max_val = freq_sum.max() #prediction label with the max probability
    index= freq_sum.argmax() #prediction label with the max probability
       
    return index #prediction
    
    
    
    
    
    