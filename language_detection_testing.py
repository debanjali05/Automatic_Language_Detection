# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Task: Implementation of a Language detection model based on N-grams

Testing the Language detection model
"""


import os
import random
import numpy as np

from data import load_dataset
from utils import checkpoint_path, lang, n
from language_modelling import predict_language_model

#reading the testing data text files in unicode
print("\t-------Loading Test Set-------")
test_set = load_dataset("Testing") #generating the train set
#test_set = random.sample(test_set,100) #random sampling of test sets (reducing the size of test set)
print("Length of Test set:", len(test_set))

print("\t-------Start Testing------")

#Loading saved model
print("Loading",n,"-gram Saved Models------->")
model = [np.load(os.path.join(checkpoint_path,l+str(n)+"gram.npy")) for l in lang] 
 
confusion_matrix = np.zeros((5,5))   
correct_prediction = 0
    
#Prediction
print("Start Predicting------->") 
for (words,label) in test_set:
    #Predicting the language
    prediction = predict_language_model(words,n,model)
    if(lang[prediction] == label):
            correct_prediction += 1
    
    pred_index = 0
    cor_index = 0
                
    for j,language in enumerate(lang):
        if(language == lang[prediction]):
            pred_index = j
        if(language == label):
            cor_index = j
                
    confusion_matrix[cor_index][pred_index] += 1
                
print("Confusion Matrix: \n",confusion_matrix)
print(n,"-gram Accuracy", correct_prediction/len(test_set)*100)
print("\t-------End Testing------")                
