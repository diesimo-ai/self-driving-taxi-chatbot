# File : main.py 
# Description : launch the main application 
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
# Afondiel  |  09.12.2020 | Creation 
# Afondiel  |  23.02.2021 | Last modification 
# Afondiel  |  02.03.2022 | move main into root directory 

import os
import sys
import json
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from module.chat_launch import chat_start
from module.data_processing import nlp_functions
from module.chatclassifier.data_collection import data_prep
from module.chatclassifier.chat_dataset import ChatDataset
from module.chatclassifier.chat_model import ChatNeuralNet
from module.chatclassifier.chat_classifier import ChatClassifier


""" 
Main : This file is the main module
"""
# include file access outside src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__' :
    # starting the app
    print("starting the Chat")

    """ PRE TRAINING THE MODEL
        # create chat classifier object
        # chatclassifier = ChatClassifier()
        # training the model
        # chatclassifier.execute()
        # save the model to a file
        # chatclassifier.save_model()
    """

    # launching the chat
    chat_start()

    # leaving the Chat
    print("leaving the Chat...")
