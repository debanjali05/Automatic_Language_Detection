# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Task: Implementation of a Language detection model based on N-grams

Testing the Language detection model
"""


import os
import numpy as np

from data import load_dataset
from utils import checkpoint_path, lang
from language_modelling import predict_language_model

#reading the testing data text files in unicode
print("\t-------Loading Test Set-------")
test_set = load_dataset("Testing") #generating the train set
print("Length of Test set:", len(test_set))

print("\t-------Start Testing------")
 
for n in range(2,3):
    
    #Loading saved model
    print("Loading",n,"-gram Saved Models------->")
    model = [np.load(os.path.join(checkpoint_path,l+str(n)+"gram.npy")) for l in lang] 
 
    """
    #Loading saved model
    print("Loading saved",n,"-gram Language Detection model------->")
    classifier_name = "classifier" + str(n)+".pickle"
    f = open(os.path.join(checkpoint_path,classifier_name), 'rb')
    classifier = pickle.load(f)
    f.close()
    """
    confusion_matrix = np.zeros((5,5))   
    correct_prediction = 0
    
    #Prediction
    print("Start Predicting------->") 
    for (words,label) in test_set:
        
        #Generating feature sets
        #feature_sets = generating_features(words,n)
        
        #Predicting the language
        prediction = predict_language_model(words,n,model)
        
        if(prediction == label):
            correct_prediction += 1
        
        pred_index = 0
        cor_index = 0
                
        for j,language in enumerate(lang):
            if(language == prediction):
                pred_index = j
            if(language == label):
                cor_index = j
                
        confusion_matrix[cor_index][pred_index] += 1
                
    print("Confusion Matrix",confusion_matrix)
    print(n,"-gram Accuracy", correct_prediction/len(test_set)*100)
print("\t-------End Testing------")                
