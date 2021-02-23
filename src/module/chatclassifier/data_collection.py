# File : data_collection.py 
# Description : intended for dataset preparation  
#
# Libraries requirements : 
#   <os> : standard library
#		- Provides a portable way of using operating system dependent functionality.
#   <sys> : standard library
#		- provides access to some variables used or maintained by the interpreter and 
#         to functions that interact strongly with the interpreter.
#   <json> : standard library
#		- Provides functionalites for data storing and serialization
#   <nltk> : standard library
#		- Provides comprehesive natural language toolkit, with models and training data in multiples languages
#   <torch> : standard library
#		- Provides multidimensional arrays and linear algebra tools, optimized for speed
#   <numpy> : standard library
#		- Provides multidimensional arrays and linear algebra tools, optimized for speed
#
# File history :
# Afondiel  |  23.02.2021 | Last modification 

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

from module.data_processing.nlp_functions import tokenize, stem, clean_data, bag_of_words

def data_prep():
    """
        - DESC : Collect training data
        - INPUT : 
        - OUTPUT : 
    """
    json_path = '/mnt/d/Lab/Project/self-driving-taxi-chatbot/data/serialization/chat.json'
    with open(json_path, 'r') as f:
        conversation = json.load(f)

    all_words = []
    tags = []
    xy = []
    bags_of_words = []

    # performing actions for each intent
    for intent in conversation['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            pattern_tok = tokenize(pattern)
            all_words.extend(pattern_tok)
            xy.append((pattern_tok, tag))

    # cleaning ponctuation charaters
    all_words = clean_data(words=all_words, tok=pattern_tok)
    # clean dublitaed elements
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # create training data for the model
    X_train = []
    y_train = []
    # create bag of words for input training data 
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(tokenized_setence=pattern_sentence, all_words=all_words)
        X_train.append(bag)
        # create output label for the training data
        label = tags.index(tag)
        # used for cross entropy
        y_train.append(label) 
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # print(y_train)
    return X_train, y_train, tags, all_words
