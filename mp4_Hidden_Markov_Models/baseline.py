# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

from collections import Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    output = []
    word_tag_dict = {}          # for seen words in training data
    tag_count = Counter()       # for unseen words in test data

    # counts in training data 
    for sentence in train:
        for wt_pair in sentence:
                word, tag = wt_pair
                # for tag counts
                tag_count[tag] += 1

                # for newly seen words
                if word not in word_tag_dict:
                        word_tag_dict[word] = Counter()
                # for all words, increment counts
                word_tag_dict[word][tag] += 1

    # find the tag with maximum counts for unseen words
    tag_max = (tag_count.most_common())[0][0]
    # print(tag_max)

    for sentence in test:
        sentence_wt_pair = []
        for word in sentence:
                if word in word_tag_dict:
                        pdt_tag = max(((word_tag_dict[word]).keys()), key = ((word_tag_dict[word]).get) )
                else:
                        pdt_tag = tag_max
                sentence_wt_pair.append((word, pdt_tag))
        output.append(sentence_wt_pair)

    return output