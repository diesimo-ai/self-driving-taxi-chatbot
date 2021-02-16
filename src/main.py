
# File : main.py 
#
# Description : This is IA chatbot application for self-driving taxi using NLP (Natural Language Processing ) \
#                   and Deep Learning
#
# Libraries requirements : 
#   <numpy> : standard library
#		- Provides multidimensional arrays and linear algebra tools, optimized for speed
#
# File history :
# Afondiel  |  09.12.2020 | Creation 

import os
import json
import nltk
import functions


""" 
Main : This file is the main module
"""

if __name__ == '__main__' :
    print("start module")

    test_list = "I am going to be a Legend!"
    tokenize_list = []
    all_words = []

    tokenize_list = functions.tokenize(test_list)
    for item in tokenize_list:
        all_words.append(functions.stem(item))

    print(tokenize_list)
    print(all_words)
    
    
