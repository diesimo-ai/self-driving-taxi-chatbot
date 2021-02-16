# file name : functions.py

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

def clean_data():
    """
        - DESC : exclude pontuation characters
        - INPUT : 
        - OUTPUT : 
    """
    pass

def bag_of_words(tokenize_setence, all_words):
    """
        - DESC : 
        - INPUT : 
        - OUTPUT : 
    """
    pass