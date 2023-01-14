# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021
# Modified by Mohit Goyal (mohit@illinois.edu) on 01/16/2022
"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator


# def compute_tf_idf(train_set, train_labels, dev_set):
#     """
#     train_set - List of list of words corresponding with each movie review
#     example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
#     Then train_set := [['like','this','movie'], ['i','fall','asleep']]

#     train_labels - List of labels corresponding with train_set
#     example: Suppose I had two reviews, first one was positive and second one was negative.
#     Then train_labels := [1, 0]

#     dev_set - List of list of words corresponding with each review that we are testing on
#               It follows the same format as train_set

#     Return: A list containing words with the highest tf-idf value from the dev_set documents
#             Returned list should have same size as dev_set (one word from each dev_set document)
#     """



#     # TODO: Write your code here
    


#     # return list of words (should return a list, not numpy array or similar)
#     return []

def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """


    # TODO: Write your code here
    word_dict = {}

    best_word = []

    for i in range(len(train_set)):
        #count_doc = Counter(train_set[i])
        freq = {}
        for k in range(len(train_set[i])):
                if train_set[i][k] not in freq.keys():
                        freq[train_set[i][k]] = 1

        #for j in count_doc.elements():
        for j in freq:
                if j in word_dict.keys():
                        word_dict[j] += 1
                else:
                        word_dict[j] = 1

    for i in range(len(dev_set)):
        #dev_dict = {}
        #count_dev_doc = Counter(dev_set[i])
        count_dev_doc = {}
        for j in range(len(dev_set[i])):
                if dev_set[i][j] in count_dev_doc:
                        count_dev_doc[dev_set[i][j]] += 1
                else:
                        count_dev_doc[dev_set[i][j]] = 1

        #total_val = sum(count_dev_doc.values())
        total_val = len(dev_set[i])
        max_val = 0
        for j in count_dev_doc.keys():
                #tf = count_dev_doc[j]/total_val
                if j in word_dict.keys():
                        dev_tfidf = (count_dev_doc[j]/total_val)*np.log(len(train_set)/(1+word_dict[j]))
                else:
                        dev_tfidf = (count_dev_doc[j]/total_val)*np.log(len(train_set))
                # if j.isalpha():
                #         dev_dict[j] = tf*idf
                # else:
                #         dev_dict[j] = -1
                #dev_tfidf = tf*idf
                if dev_tfidf > max_val:
                        max_val = dev_tfidf
                        max_key = j

        # max_key = []
        # for key, value in dev_dict.items():
        #         if value == max(dev_dict.values()):
        #                 max_key.append(key)
        #                 # break
        # print("max_key:", max_key)

        # if len(max_key) > 1:
        #         #for word in dev_set[i]:
        #         for j in range(len(dev_set[i])):
        #                 if dev_set[i][j] in max_key:
        #                         best_word.append(dev_set[i][j])
        #                         break
        # else:
        #         best_word.append(max_key[0])
        best_word.append(max_key)

    #print(len(best_word))
    #print(len(dev_set))

#print(best_word)


    # return list of words (should return a list, not numpy array or similar)
    return best_word