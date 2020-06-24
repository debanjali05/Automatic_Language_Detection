# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:29:56 2020

@author: debanjalibiswas

Task: Implementation of a Language detection model based on N-grams

Constants
"""

#We consider 5 languages from the Corpora dataset
lang = ["french" ,"english","german","italian","dutch"]



#Dataset path (Update correct path )
train_path = "Data/Train"
train_lang_path = ["fra_newscrawl_2014_30K/fra_newscrawl_2014_30K-sentences.txt","eng_wikipedia_2016_30K/eng_wikipedia_2016_30K-sentences.txt","deu_newscrawl_2017_30K/deu_newscrawl_2017_30K-sentences.txt","ita_wikipedia_2016_30K/ita_wikipedia_2016_30K-sentences.txt","nld_wikipedia_2016_30K/nld_wikipedia_2016_30K-sentences.txt"]

#Testing Dataset path (Update correct path )
test_path = "Data/Test"
test_lang_path = ["fra_newscrawl_2014_10K/fra_newscrawl_2014_10K-sentences.txt","eng_wikipedia_2016_10K/eng_wikipedia_2016_10K-sentences.txt","deu_newscrawl_2017_10K/deu_newscrawl_2017_10K-sentences.txt","ita_wikipedia_2016_10K/ita_wikipedia_2016_10K-sentences.txt","nld_wikipedia_2016_10K/nld_wikipedia_2016_10K-sentences.txt"]

#path to store the model checkpoints 
checkpoint_path = "checkpoints" 

