# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Task: Implementation of a Language detection model based on N-grams

Reading Dataset
"""

import re
import os
import string
import codecs

from utils import lang, train_path, train_lang_path, test_path, test_lang_path

def preprocessing(line):
    """
        data preprocessing step: 
            1) Convert to lowercase
            2) Remove line number, digits, punctuations and extra spaces
    
        line: each sentence from the dataset (string)
    """
    
    translate_table = dict((ord(char), None) for char in string.punctuation)
    
    line = line.lower() #converting the text to lowercase 
    line = re.sub(r"\d+", "", line) #removing any digits present in the text
    line = line.translate(translate_table)  #removing all punctuations
    line = re.sub(' +',' ',line) #removing extra spaces
    line = line.split()[1:] #removing the line numbers from the text files
     
    return line #preprocessed text

def build_dataset(path,language):
    """
        building the dataset with language label
    
        path: path to the text
        language: language label (string)
    """
    
    language_set = []
    words_all = []
    
    #Reading the training data text files in unicode
    with codecs.open(path,"r","utf-8") as filep:
         
        for i,line in enumerate(filep): 
            line = preprocessing(line) #preprocessing on data
            df = (line,language) #adding language label
            language_set.append(df)
            words_all += line
            words_all.append(" ") #append a space after every line
          
    return language_set, words_all #individual language set and list of words

def load_dataset(mode = "Training"):
    """
        creating the train and test sets
    
        mode:"Training"/"Testing" (string)
    """
    
    data = []
    
    if(mode == "Training"): 
        
        #Creating train set
        for i,l in enumerate(lang):
            
            path = os.path.join(train_path,train_lang_path[i]) #path to train text
            lang_set,_ = build_dataset(path,l) #generating dataset for individual language
            data = data + lang_set # final train set
            
    elif(mode == "Testing"): 
        
        #Creating test set
        for i,l in enumerate(lang): 
            
            path = os.path.join(test_path,test_lang_path[i]) #path to test text
            lang_set,_ = build_dataset(path,l)  #generating dataset for individual language
            data = data + lang_set #final test set
    
    else:
        print("Incorrect mode")
        
    return data #train/test set



 