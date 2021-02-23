# File : chat_model.py 
# Description : training model 
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

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ChatNeuralNet(nn.Module):
    """
    - DESC : Feed Forward Network(FFN) : 
        - 1 input layer : get bag of words data
        - 1 hidden layer : numbers of patterns
        - 1 output layer : numbers of class
        - Softmax function :  for probability prediction output value
        - accuracy rate (%) ? 
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        - DESC : init constructor
        - INPUT : 
            - self : this object
            - input_size 
            - hidden_size
            - output_size
        - OUTPUT :  return score prediction
        """
        super(ChatNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        # activation function 
        self.relu = nn.ReLU()
    
    def forward (self, x):
        """
        - DESC : forward model
        - INPUT : 
            - self : this object
            - x : dataset to be trained
        - OUTPUT :  return score prediction
        """
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation no and no softmax
        return out 