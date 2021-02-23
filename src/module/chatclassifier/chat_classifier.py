# File : chat_classifier.py 
# Description : implements the classification model  
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
# Afondiel  |  23.02.2021 | last modification 

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from module.data_processing import nlp_functions
from module.chatclassifier.data_collection import data_prep
from module.chatclassifier.chat_dataset import ChatDataset
from module.chatclassifier.chat_model import ChatNeuralNet

class ChatClassifier():
    """
    - DESC : Collect training data
    """
    def __init__(self):
        """
        - DESC : Collect training data
        - INPUT : self (this object)
        - OUTPUT : retrun score 
        """
        # Hyperparameters
        self.batch_size = 8
        self.learning_rate = 0.001
        self.num_epochs = 1000

        # get training data
        self.x_train, self.y_train, self.tags, self.all_words = data_prep()
        # creata chatdataset object
        self.chatdataset = ChatDataset(
                                    X_train = self.x_train, 
                                    y_train = self.y_train)

        # loading dataset for the model
        self.train_loader = DataLoader(dataset=self.chatdataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        # check if there is GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create the model
        self.model = ChatNeuralNet(
                            input_size = len(self.x_train[0]), 
                            hidden_size = 8, 
                            output_size = len(self.tags)).to(self.device)

    def execute(self):
        """
        - DESC : 
        - INPUT : self (this object)
        - OUTPUT : 
        """
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # training loop
        for epoch in range(self.num_epochs):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(self.device)
                # forward
                outputs = self.model(words)
                loss = criterion(outputs, labels)

                # backward and optimizer gradient and step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch +1) % 100 == 0:
                print(f'epoch {epoch + 1}/{self.num_epochs}, loss={loss.item():.4f}')

        print(f'final loss, loss={loss.item():.4f}')
    
    def save_model(self):
        """
        - DESC : 
        - INPUT : self (this object)
        - OUTPUT :  
        """
        # model parameters to be saved
        data = {
                "model_state": self.model.state_dict(),
                "input_size":  len(self.x_train[0]),
                "hidden_size": 8,
                "output_size": len(self.tags),
                "all_words" : self.all_words,
                "tags" : self.tags
            }

        # Saving the model
        FILE = "/mnt/d/Lab/Project/self-driving-taxi-chatbot/data/model/data.pth"
        torch.save(data, FILE)

        print(f'training complete, file saved to {FILE}')
