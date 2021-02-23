# File : nlp_functions.py 
# Description : nlp tools
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

import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):

    """
        - DESC : 
        - INPUT : 
        - OUTPUT : 
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    
    """
        - DESC : 
        - INPUT : 
        - OUTPUT : 
    """
    return stemmer.stem(word.lower())

def clean_data(words, tok):
    """
        - DESC : 
        - INPUT : 
        - OUTPUT : 
    """
    ignore_words = ['?', '!', '.', ',', ')', ':']
    words = [stem(tok) for tok in words if tok not in ignore_words]
    return words

def bag_of_words(tokenized_setence, all_words):
    """
        - DESC : 
            - if tok is in all_words :
            we put '1' in the index of token in the all_words list
            other wise '0' 
            tok : ['see', 'you', 'soon']
            all_words : ['adress', 'are', 'destin', 'enter', 'good', 'goodby', 'hello', 'how', 'morn', 'music', 'pl', 'pleas', 'see', 'some', 'soon', 'you', 'your']
            bags_of_words : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]

        - INPUT : 
        - OUTPUT : 
    """
    # solutiion 1 : 
    # initial a new list with same size as all_words to zero
    # bag = [0]*len(all_words)
    # # print(bag_of_word)
    # for idx, item in enumerate(all_words):
    #     for tok in tokenized_setence:
    #         if tok == item  :
    #             bag[idx] = 1

    # solutiion 2 : using numpy
    # stemming the tokenized words
    tokenized_setence = [stem(w) for w in tokenized_setence]
    # create an array of zero
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_setence:
            bag[idx] = 1.0

    return bag