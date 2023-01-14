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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

from collections import Counter
import math

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Calculate initial probabilities
    # unique word and tag sets
    # tag occurrences in the beginning of sentence
    # tag pair counts
    # word/tag pair counts
    unique_word = set()
    unique_tag = set()
    tag_count = Counter()
    word_count = Counter()
    begin_tag_count = Counter()
    tag_pair_count = Counter()
    word_tag_count = Counter()
    tag_hapax_dict = Counter()

    laplace_smooth_k = 1e-4

    for sentence in train:
        # tag occurrences in the beginning of sentence 
        temp_begin_tag = sentence[0][1]
        begin_tag_count[temp_begin_tag] += 1

        for word_tag_pair in sentence:
                word, tag = word_tag_pair
                #unique sets
                unique_word.add(word)
                unique_tag.add(tag)
                tag_count[tag] += 1
                word_count[word] += 1

                if tag not in tag_hapax_dict:
                        tag_hapax_dict[tag] = Counter()
                tag_hapax_dict[tag][word] += 1
                # increment all word-tag pair counts
                word_tag_count[word_tag_pair] += 1

    begin_tag_dict = dict(begin_tag_count)
    word_tag_dict = dict(word_tag_count)

    # for tag pairs
    for sentence in train:
        for i in range(1,len(sentence)):
                cur_tag = sentence[i][1]
                prev_tag = sentence[i-1][1]
                tag_pair = (prev_tag, cur_tag)
                tag_pair_count[tag_pair] += 1 
    tag_pair_dict = dict(tag_pair_count)

    # calculate tag probabilities of start in sentence
    begin_tag_prob = {}
    for begin_tag, count in begin_tag_dict.items():
            begin_tag_prob[begin_tag] = math.log((count+laplace_smooth_k) / (len(train) + laplace_smooth_k*len(unique_tag)))
    begin_unseen_tag_prob = math.log((laplace_smooth_k) / (len(train) + laplace_smooth_k*len(unique_tag)))
    begin_tag_prob[-1] = begin_unseen_tag_prob


    # Deal with hapax words
    hapax_words = []
    for word, word_val in word_count.items():
        if word_val == 1:
                hapax_words.append(word)

    tag_hapax_prob = {}
    for tag in unique_tag:
        tag_hapax_sum = 0
        for hapax in hapax_words:
                tag_hapax_sum += tag_hapax_dict[tag][hapax]
        tag_hapax_prob[tag] = (tag_hapax_sum + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))

    # calculate tag-word probabilities
    tag_word_prob = {}
    for tag in unique_tag:
        laplace_smooth_hapax = tag_hapax_prob[tag]*laplace_smooth_k
        for word in unique_word:
                temp_pair = (word, tag)
                if temp_pair in word_tag_dict:
                        tag_word_prob[temp_pair] = math.log((word_tag_dict[temp_pair] + laplace_smooth_k)/ (tag_count[tag] + laplace_smooth_k*(len(unique_word)+1)))
        unseen_wt_pair = (-1, tag)
        tag_word_prob[unseen_wt_pair] = math.log((laplace_smooth_hapax) / (tag_count[tag] + laplace_smooth_hapax*(len(unique_word)+1)))

    # calculate tag-tag transition probabilities
    tag_trans_prob = {}
    tag_trans_total = 0
    for tag in unique_tag:
            tag_trans_count = 0
            for post_tag in unique_tag:
                if (tag, post_tag) in tag_pair_dict:
                        tag_trans_count += tag_pair_dict[(tag, post_tag)]
                        tag_trans_total += tag_pair_dict[(tag, post_tag)]
            for post_tag in unique_tag:
                if (tag, post_tag) in tag_pair_dict:
                        tag_trans_prob[(tag, post_tag)] = math.log((tag_pair_dict[(tag, post_tag)] + laplace_smooth_k)/ (tag_trans_count + laplace_smooth_k*(len(unique_tag))))
                else:
                        tag_trans_prob[(tag, post_tag)] = math.log((laplace_smooth_k)/ (tag_trans_count + laplace_smooth_k*(len(unique_tag))))
    tag_trans_unseen_prob = math.log((laplace_smooth_k)/ (tag_trans_total + laplace_smooth_k*(len(unique_tag))))
    tag_trans_prob[-1] = tag_trans_unseen_prob

    output = []
    for sentence in test:

        # initialize weight and backpointer for each position for each tag
        weight_matrix = []
        backptr = []
        for i in range(len(sentence)):
                position_dict = {tag:0 for tag in unique_tag}
                position_ptr_dict = {tag:None for tag in unique_tag}
                weight_matrix.append(position_dict)
                backptr.append(position_ptr_dict)

        sentence_predict = forward_trellis(sentence, weight_matrix, backptr, begin_tag_prob, tag_word_prob, tag_trans_prob)
        output.append(sentence_predict)
    
    return output

def forward_trellis(sentence, weight_matrix, backptr, begin_tag_prob, tag_word_prob, tag_trans_prob):
    sentence_predict = []

    # initialize start prob for the first tag
    for tag in weight_matrix[0].keys():
        if tag in begin_tag_prob:
                tag_prior = begin_tag_prob[tag]
        else:
                tag_prior = begin_tag_prob[-1]
        
        temp_wt_pair = (sentence[0], tag)
        if temp_wt_pair in tag_word_prob:
                b = tag_word_prob[temp_wt_pair]
        else:
                b = tag_word_prob[(-1, tag)]

        weight_matrix[0][tag] = tag_prior + b

    # forward
    for i in range(1, len(sentence)):
        for tag in weight_matrix[i].keys():
                max_prob = float("-inf")
                max_prev_tag = ""

                temp_tag_word_pair = (sentence[i], tag)
                if temp_tag_word_pair in tag_word_prob:
                        b = tag_word_prob[temp_tag_word_pair]
                else:
                        b = tag_word_prob[(-1, tag)]

                for prev_tag in weight_matrix[i-1].keys():
                        temp_tag_pair = (prev_tag, tag)
                        if temp_tag_pair in tag_trans_prob:
                                a = tag_trans_prob[temp_tag_pair]
                        else:
                                a = tag_trans_prob[-1]

                        temp_weight = a + b + weight_matrix[i-1][prev_tag]
                        if temp_weight > max_prob:
                                max_prob = temp_weight
                                max_prev_tag = prev_tag
                        
                weight_matrix[i][tag] = max_prob
                backptr[i][tag] = max_prev_tag

    end_idx = len(sentence)-1
    reverse_predict = []
    max_end_tag = max(weight_matrix[end_idx], key = weight_matrix[end_idx].get)
    while (max_end_tag != None) and (end_idx >= 0 ) :
        add_pair = (sentence[end_idx], max_end_tag)
        reverse_predict.append(add_pair)
        max_end_tag = backptr[end_idx][max_end_tag]
        end_idx -= 1

    end_idx = len(sentence)-1
    while end_idx >= 0:
        sentence_predict.append(reverse_predict[end_idx])
        end_idx -= 1

    return sentence_predict