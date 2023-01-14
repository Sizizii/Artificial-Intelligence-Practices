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
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

from collections import Counter
import math

def viterbi_ec(train, test):
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

    number_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
        'hundred', 'thousand', 'million', 'billion']

    # Deal with hapax words
    hapax_words = []
    for word, word_val in word_count.items():
        if word_val == 1:
                hapax_words.append(word)

    tag_hapax_prob = {}
    tag_hapax_special = {}
    for tag in unique_tag:
        tag_hapax_sum = 0
        tag_hapax_ly = 0
        tag_hapax_ing = 0
        tag_hapax_ive = 0
        tag_hapax_ble = 0
        tag_hapax_ed = 0
        tag_hapax_er = 0
        tag_hapax_es = 0
        tag_hapax_tion = 0
        tag_hapax_ty = 0
        tag_hapax_nal = 0
        tag_hapax_ful = 0
        tag_hapax_tor = 0
        tag_hapax_age = 0
        tag_hapax_digit = 0
        tag_hapax_num = 0
        for hapax in hapax_words:
                if hapax.endswith("ly"):
                        tag_hapax_ly += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("ing"):
                        tag_hapax_ing += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("ive"):
                        tag_hapax_ive += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("ble"):
                        tag_hapax_ble += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("ed"):
                        tag_hapax_ed += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("er"):
                        tag_hapax_er += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("es"):
                        tag_hapax_es += tag_hapax_dict[tag][hapax]
                # elif hapax.endswith("ist"):
                        # tag_hapax_ist += tag_hapax_dict[tag][hapax]
                # elif hapax.endswith("tion"):
                        # tag_hapax_tion += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("ty"):
                        tag_hapax_ty += tag_hapax_dict[tag][hapax]
                elif hapax.endswith("age"):
                        tag_hapax_age += tag_hapax_dict[tag][hapax]
                elif hapax.isdigit():
                        tag_hapax_digit += tag_hapax_dict[tag][hapax]
                elif (any(num in number_list for num in hapax)  and hapax.isalpha()):
                        tag_hapax_num += tag_hapax_dict[tag][hapax]
                # elif hapax.endswith("ism"):
                        # tag_hapax_ism += tag_hapax_dict[tag][hapax]
                # elif hapax.endswith("nal"):
                        # tag_hapax_nal += tag_hapax_dict[tag][hapax]
                # elif (hapax.endswith("tor") or hapax.endswith("tors")):
                        # tag_hapax_tor += tag_hapax_dict[tag][hapax]
                # elif hapax.endswith("ful"):
                        # tag_hapax_ful += tag_hapax_dict[tag][hapax]
                # elif hapax.startswith("ab"):
                        # tag_hapax_ab += tag_hapax_dict[tag][hapax]
                else:
                        tag_hapax_sum += tag_hapax_dict[tag][hapax]
        tag_hapax_prob[tag] = (tag_hapax_sum + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ly", tag)] = (tag_hapax_ly + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ing", tag)] = (tag_hapax_ing + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ive", tag)] = (tag_hapax_ive + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ble", tag)] = (tag_hapax_ble + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ed", tag)] = (tag_hapax_ed + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("er", tag)] = (tag_hapax_er + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("es", tag)] = (tag_hapax_es + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("ist", tag)] = (tag_hapax_ist + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("tion", tag)] = (tag_hapax_tion + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("ty", tag)] = (tag_hapax_ty + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("age", tag)] = (tag_hapax_age + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("digit", tag)] = (tag_hapax_digit + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        tag_hapax_special[("num", tag)] = (tag_hapax_num + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("ment", tag)] = (tag_hapax_ment + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("ism", tag)] = (tag_hapax_ism + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("nal", tag)] = (tag_hapax_nal + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("ab", tag)] = (tag_hapax_ab + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("ful", tag)] = (tag_hapax_ful + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))
        # tag_hapax_special[("tor", tag)] = (tag_hapax_tor + laplace_smooth_k) / (len(hapax_words) + laplace_smooth_k*len(unique_tag))

    # calculate tag-word probabilities
    tag_word_prob = {}
    for tag in unique_tag:
        laplace_smooth_hapax = tag_hapax_prob[tag]*laplace_smooth_k
        laplace_ly = tag_hapax_special[("ly", tag)]*laplace_smooth_k
        laplace_ing = tag_hapax_special[("ing", tag)]*laplace_smooth_k
        laplace_ive = tag_hapax_special[("ive", tag)]*laplace_smooth_k
        laplace_ble = tag_hapax_special[("ble", tag)]*laplace_smooth_k
        laplace_ed = tag_hapax_special[("ed", tag)]*laplace_smooth_k
        laplace_er = tag_hapax_special[("er", tag)]*laplace_smooth_k
        laplace_es = tag_hapax_special[("es", tag)]*laplace_smooth_k
        # laplace_ist = tag_hapax_special[("ist", tag)]*laplace_smooth_k
        # laplace_tion = tag_hapax_special[("tion", tag)]*laplace_smooth_k
        laplace_ty = tag_hapax_special[("ty", tag)]*laplace_smooth_k
        laplace_age = tag_hapax_special[("age", tag)]*laplace_smooth_k
        laplace_digit = tag_hapax_special[("digit", tag)]*laplace_smooth_k
        laplace_num = tag_hapax_special[("num", tag)]*laplace_smooth_k
        # laplace_ment = tag_hapax_special[("ment", tag)]*laplace_smooth_k
        # laplace_ism = tag_hapax_special[("ism", tag)]*laplace_smooth_k
        # laplace_ab = tag_hapax_special[("ab", tag)]*laplace_smooth_k
        # laplace_nal = tag_hapax_special[("nal", tag)]*laplace_smooth_k
        # laplace_ful = tag_hapax_special[("ful", tag)]*laplace_smooth_k
        # laplace_tor = tag_hapax_special[("tor", tag)]*laplace_smooth_k
        for word in unique_word:
                temp_pair = (word, tag)
                if temp_pair in word_tag_dict:
                        tag_word_prob[temp_pair] = math.log((word_tag_dict[temp_pair] + laplace_smooth_k)/ (tag_count[tag] + laplace_smooth_k*(len(unique_word)+1)))
        unseen_wt_pair = (-1, tag)
        special_pair_ly = ("ly", tag)
        special_pair_ing = ("ing", tag)
        special_pair_ive = ("ive", tag)
        special_pair_ble = ("ble", tag)
        special_pair_ed = ("ed", tag)
        special_pair_er = ("er", tag)
        special_pair_es = ("es", tag)
        # special_pair_ab = ("ab", tag)
        # special_pair_tion = ("tion", tag)
        special_pair_ty = ("ty", tag)
        special_pair_age = ("age", tag)
        special_pair_digit = ("digit", tag)
        special_pair_num = ("num", tag)
        # special_pair_ment = ("ment", tag)
        # special_pair_ism = ("ism", tag)
        # special_pair_nal = ("nal", tag)
        # special_pair_ful = ("ful", tag)
        # special_pair_tor = ("tor", tag)
        # special_pair_ist = ("ist", tag)
        tag_word_prob[unseen_wt_pair] = math.log((laplace_smooth_hapax) / (tag_count[tag] + laplace_smooth_hapax*(len(unique_word)+1)))
        tag_word_prob[special_pair_ly] = math.log((laplace_ly) / (tag_count[tag] + laplace_ly*(len(unique_word)+1)))
        tag_word_prob[special_pair_ing] = math.log((laplace_ing) / (tag_count[tag] + laplace_ing*(len(unique_word)+1)))
        tag_word_prob[special_pair_ive] = math.log((laplace_ive) / (tag_count[tag] + laplace_ive*(len(unique_word)+1)))
        tag_word_prob[special_pair_ble] = math.log((laplace_ble) / (tag_count[tag] + laplace_ble*(len(unique_word)+1)))
        tag_word_prob[special_pair_ed] = math.log((laplace_ed) / (tag_count[tag] + laplace_ed*(len(unique_word)+1)))
        tag_word_prob[special_pair_er] = math.log((laplace_er) / (tag_count[tag] + laplace_er*(len(unique_word)+1)))
        tag_word_prob[special_pair_es] = math.log((laplace_es) / (tag_count[tag] + laplace_es*(len(unique_word)+1)))
        # tag_word_prob[special_pair_tion] = math.log((laplace_tion) / (tag_count[tag] + laplace_tion*(len(unique_word)+1)))
        tag_word_prob[special_pair_ty] = math.log((laplace_ty) / (tag_count[tag] + laplace_ty*(len(unique_word)+1)))
        tag_word_prob[special_pair_age] = math.log((laplace_age) / (tag_count[tag] + laplace_age*(len(unique_word)+1)))
        tag_word_prob[special_pair_digit] = math.log((laplace_digit) / (tag_count[tag] + laplace_digit*(len(unique_word)+1)))
        tag_word_prob[special_pair_num] = math.log((laplace_num) / (tag_count[tag] + laplace_num*(len(unique_word)+1)))
        # tag_word_prob[special_pair_ment] = math.log((laplace_ment) / (tag_count[tag] + laplace_ment*(len(unique_word)+1)))
        # tag_word_prob[special_pair_ism] = math.log((laplace_ism) / (tag_count[tag] + laplace_ism*(len(unique_word)+1)))
        # tag_word_prob[special_pair_nal] = math.log((laplace_nal) / (tag_count[tag] + laplace_nal*(len(unique_word)+1)))
        # tag_word_prob[special_pair_ab] = math.log((laplace_ab) / (tag_count[tag] + laplace_ab*(len(unique_word)+1)))
        # tag_word_prob[special_pair_ful] = math.log((laplace_ful) / (tag_count[tag] + laplace_ful*(len(unique_word)+1)))
        # tag_word_prob[special_pair_tor] = math.log((laplace_tor) / (tag_count[tag] + laplace_tor*(len(unique_word)+1)))
        # tag_word_prob[special_pair_ist] = math.log((laplace_ist) / (tag_count[tag] + laplace_ist*(len(unique_word)+1)))


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

        sentence_predict = forward_trellis(sentence, weight_matrix, backptr, begin_tag_prob, tag_word_prob, tag_trans_prob, unique_word)
        output.append(sentence_predict)
    
    return output

def forward_trellis(sentence, weight_matrix, backptr, begin_tag_prob, tag_word_prob, tag_trans_prob, unique_word):
    sentence_predict = []

    number_list = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
        'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
        'hundred', 'thousand', 'million', 'billion']

    # initialize start prob for the first tag
    for tag in weight_matrix[0].keys():
        if tag in begin_tag_prob:
                tag_prior = begin_tag_prob[tag]
        else:
                tag_prior = begin_tag_prob[-1]
        
        temp_wt_pair = (sentence[0], tag)
        temp_word = sentence[0]
        if temp_wt_pair in tag_word_prob:
                b = tag_word_prob[temp_wt_pair]
        else:
                if temp_word.endswith("ly"):
                        b = tag_word_prob[("ly", tag)]
                elif temp_word.endswith("ing"):
                        b = tag_word_prob[("ing", tag)]
                elif temp_word.endswith("ive"):
                        b = tag_word_prob[("ive", tag)]
                elif temp_word.endswith("ble"):
                        b = tag_word_prob[("ble", tag)]
                elif temp_word.endswith("ed"):
                        b = tag_word_prob[("ed", tag)]
                elif temp_word.endswith("er"):
                        b = tag_word_prob[("er", tag)]
                elif temp_word.endswith("es"):
                        b = tag_word_prob[("es", tag)]
                # elif temp_word.endswith("tion"):
                        # b = tag_word_prob[("tion", tag)]
                elif temp_word.endswith("ty"):
                        b = tag_word_prob[("ty", tag)]
                elif temp_word.endswith("age"):
                        b = tag_word_prob[("age", tag)]
                elif temp_word.isdigit():
                        b = tag_word_prob[("digit", tag)]
                elif (any(num in number_list for num in temp_word)  and temp_word.isalpha()):
                        b = tag_word_prob[("num", tag)]
                # elif temp_word.endswith("ism"):
                        # b = tag_word_prob[("ism", tag)]
                # elif temp_word.endswith("nal"):
                        # b = tag_word_prob[("nal", tag)]
                # elif temp_word.endswith("ful"):
                        # b = tag_word_prob[("ful", tag)]
                else:
                        b = tag_word_prob[(-1, tag)]

        weight_matrix[0][tag] = tag_prior + b

    # forward
    for i in range(1, len(sentence)):
        for tag in weight_matrix[i].keys():
                max_prob = float("-inf")
                max_prev_tag = ""

                temp_tag_word_pair = (sentence[i], tag)
                temp_word = sentence[i]
                if temp_tag_word_pair in tag_word_prob:
                        b = tag_word_prob[temp_tag_word_pair]
                else:
                        if temp_word.endswith("ly"):
                                b = tag_word_prob[("ly", tag)]
                        elif temp_word.endswith("ing"):
                                b = tag_word_prob[("ing", tag)]
                        elif temp_word.endswith("ive"):
                                b = tag_word_prob[("ive", tag)]
                        elif temp_word.endswith("ble"):
                                b = tag_word_prob[("ble", tag)]
                        elif temp_word.endswith("ed"):
                                b = tag_word_prob[("ed", tag)]
                        elif temp_word.endswith("er"):
                                b = tag_word_prob[("er", tag)]
                        elif temp_word.endswith("es"):
                                b = tag_word_prob[("es", tag)]
                        # elif temp_word.endswith("tion"):
                                # b = tag_word_prob[("tion", tag)]
                        elif temp_word.endswith("ty"):
                                b = tag_word_prob[("ty", tag)]
                        elif temp_word.endswith("age"):
                                b = tag_word_prob[("age", tag)]
                        elif temp_word.isdigit():
                                b = tag_word_prob[("digit", tag)]
                        elif (any(num in number_list for num in temp_word)  and temp_word.isalpha()):
                                b = tag_word_prob[("num", tag)]
                        # elif temp_word.endswith("ful"):
                                # b = tag_word_prob[("ful", tag)]
                        # elif temp_word.endswith("ism"):
                                # b = tag_word_prob[("ism", tag)]
                        # elif temp_word.endswith("nal"):
                                # b = tag_word_prob[("nal", tag)]
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

    

    pp_list = ['it','he','she']

    end_idx = len(sentence)-1
    while end_idx >= 0:
        temp_word = reverse_predict[end_idx][0]
        temp_tag = reverse_predict[end_idx][1]
        if temp_word not in unique_word:
                # if temp_word.endswith("ly"):
                        # temp_tag = 'ADV'
                # elif (temp_word.endswith("ism")) or (temp_word.endswith("isms")):
                        # temp_tag = 'NOUN'
                # elif (temp_word.endswith("tor")) or (temp_word.endswith("tors")) or (temp_word.endswith("ter")) or (temp_word.endswith("ters") or (temp_word.endswith("ties"))):
                        # temp_tag = 'NOUN'
                if temp_word.endswith("'s") and (substring  not in pp_list for substring in temp_word):
                        temp_tag = 'NOUN'
                # elif (temp_word.endswith("ment") or temp_word.endswith("ments")):
                        # temp_tag = 'NOUN'
                elif '$'in temp_word:
                        temp_tag = 'NOUN'
                elif temp_word.endswith('mous'):
                        temp_tag = 'ADJ'
                elif temp_word.endswith('ance'):
                        temp_tag = 'NOUN'
                # elif temp_word.endswith("sion"):
                        # temp_tag = 'NOUN'
                # elif (any(substring in number_list for substring in temp_word) and temp_word.endswith("th")):
                        # temp_tag = 'ADJ'
                # elif (any(substring in number_list for substring in temp_word) and temp_word.isalpha()):
                        # temp_tag = 'NUM'
                # elif any(substring in number_list for substring in temp_word):
                        # temp_tag = 'ADJ'
                # elif temp_word.endswith("ty"):
                        # temp_tag = 'NOUN'
                # elif temp_word.endswith("ies"):
                        # temp_tag = 'NOUN'
                # elif temp_word.isdigit():
                        # temp_tag = 'ADJ'
                pair = (temp_word, temp_tag)
                sentence_predict.append(pair)
        else:
                sentence_predict.append(reverse_predict[end_idx])
        end_idx -= 1
            

    return sentence_predict