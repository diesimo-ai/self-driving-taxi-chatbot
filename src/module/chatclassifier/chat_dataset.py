# File : chat_dataset.py 
# Description : formats the dataset for the model
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
# Afondiel  |  24.02.2021 | Last modification 

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    """
    - DESC : Collect training data
    """
    def __init__(self, X_train, y_train):
        """
        - DESC : Collect training data
        - INPUT : self (this object
        - OUTPUT :  
        """
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index]) 

    def __len__(self):
        return self.n_samples