# File : chat_launch.py 
# Description : implements the chat application 
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
import nltk
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from module.data_processing.nlp_functions import tokenize, stem, clean_data, bag_of_words
from module.chatclassifier.data_collection import data_prep
from module.chatclassifier.chat_dataset import ChatDataset
from module.chatclassifier.chat_model import ChatNeuralNet
from module.chatclassifier.chat_classifier import ChatClassifier

def chat_start():
    # check if there is GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # open the chat conversation
    with open('/mnt/d/Lab/Project/self-driving-taxi-chatbot/data/serialization/chat.json', 'r') as f:
        intents = json.load(f)
    # load the pre trained model
    data = torch.load('/mnt/d/Lab/Project/self-driving-taxi-chatbot/data/model/data.pth')
    # load model params
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    # init model
    model = ChatNeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # create chat launcher
    bot_name = 'Johnny'
    print("Let's chat! type 'q' to exit")

    # chat loop
    while True:
        # get input sentence
        sentence = input('You: ')
        if sentence == 'q':
            break

        # create a bag of words for the input sentence 
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        # get the predicted output
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # get the probability output using softmax
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # check if the probability is hight enough
        if prob.item() > 0.75:
        # get the response associated to the classified tag
            for intent in intents["intents"]:
                if tag == intent['tag']:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I dont understand ...")
