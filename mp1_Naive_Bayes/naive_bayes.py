# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader
"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}

    # pos_vocab = Counter()
    # neg_vocab = Counter()

    ##TODO:
    #raise RuntimeError("Replace this line with your code!")

    for i in range(len(X)):
        if y[i] == 1:
            for j in range(len(X[i])):
                if X[i][j] not in pos_vocab:
                    pos_vocab[X[i][j]] = 1
                else:
                    pos_vocab[X[i][j]] += 1
        else:
            for j in range(len(X[i])):
                if X[i][j] not in neg_vocab:
                    neg_vocab[X[i][j]] = 1
                else:
                    neg_vocab[X[i][j]] += 1
    # for i in range(len(X)):
    #     if y[i] == 1:
    #         pos_temp = Counter(X[i])
    #         pos_vocab.update(pos_temp)
    #     else:
    #         neg_temp = Counter(X[i])
    #         neg_vocab.update(neg_temp)   

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    #pos_vocab = {}
    #neg_vocab = {}

    pos_vocab = Counter()
    neg_vocab = Counter()

    ##TODO:
    #raise RuntimeError("Replace this line with your code!")

    # for i in range(len(X)):
    #     if y[i] == 1:
    #         for j in range(len(X[i]) - 1):
    #             biwords = X[i][j] + X[i][j+1]
    #             if biwords not in pos_vocab:
    #                 pos_vocab[biwords] = 1
    #             else:
    #                 pos_vocab[biwords] += 1
    #     else:
    #         for j in range(len(X[i]) - 1):
    #             biwords = X[i][j] + + X[i][j+1]
    #             if biwords not in neg_vocab:
    #                 neg_vocab[biwords] = 1
    #             else:
    #                 neg_vocab[biwords] += 1

    for i in range(len(X)):
        if y[i] == 1:
            pos_unilist = []
            for j in range(len(X[i]) - 1):
                biwords = X[i][j] + " " + X[i][j+1]
                pos_unilist.append(biwords)
            pos_temp = Counter(pos_unilist)
            pos_vocab.update(pos_temp)
            pos_temp_uni = Counter(X[i])
            pos_vocab.update(pos_temp_uni)

        else:
            neg_unilist = []
            for j in range(len(X[i]) - 1):
                biwords = X[i][j] + " " + X[i][j+1]
                neg_unilist.append(biwords)
            neg_temp = Counter(neg_unilist)
            neg_vocab.update(neg_temp)
            neg_temp_uni = Counter(X[i])
            neg_vocab.update(neg_temp_uni)

    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    #raise RuntimeError("Replace this line with your code!")

    pos_uni, neg_uni = create_word_maps_uni(train_set, train_labels)

    pos_totfreq = sum(pos_uni.values())
    neg_totfreq = sum(neg_uni.values())

    pos_term = len(pos_uni)
    neg_term = len(neg_uni)

    result = []

    for i in range(len(dev_set)):

        p_pos = np.log(pos_prior) # initialise
        p_neg = np.log(1-pos_prior)

        for j in range(len(dev_set[i])):
            temp_word = dev_set[i][j]

            if temp_word in pos_uni:
                p_posword = (pos_uni[temp_word]+laplace)/(pos_totfreq + laplace*(1+pos_term))
            else:
                p_posword = laplace/(pos_totfreq + laplace*(1+pos_term))
            p_pos += np.log(p_posword)

            if temp_word in neg_uni:
                p_negword = (neg_uni[temp_word]+laplace)/(neg_totfreq + laplace*(1+neg_term))
            else:
                p_negword = laplace/(neg_totfreq + laplace*(1+neg_term))
            p_neg += np.log(p_negword)

        if (p_pos >= p_neg) :
            result.append(1)
        else:
            result.append(0)

    return result


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    #raise RuntimeError("Replace this line with your code!")

    pos_uni, neg_uni = create_word_maps_uni(train_set, train_labels)
    pos_bi, neg_bi = create_word_maps_bi(train_set, train_labels)

    pos_totfreq_uni = sum(pos_uni.values())
    neg_totfreq_uni = sum(neg_uni.values())

    pos_totfreq_bi = sum(pos_bi.values())
    neg_totfreq_bi = sum(neg_bi.values())

    pos_term_uni = len(pos_uni)
    neg_term_uni = len(neg_uni)

    pos_term_bi = len(pos_bi)
    neg_term_bi = len(neg_bi)

    result = []

    neg_prior = 1-pos_prior

    for i in range(len(dev_set)):

        p_pos = np.log(pos_prior) # initialise
        p_neg = np.log(neg_prior)

        for k in range(len(dev_set[i])):
            temp_word_uni = dev_set[i][k]

            if temp_word_uni in pos_uni:
                p_posword_uni = (pos_uni[temp_word_uni]+unigram_laplace)/(pos_totfreq_uni + unigram_laplace*(1+pos_term_uni))
            else:
                p_posword_uni = unigram_laplace/(pos_totfreq_uni + unigram_laplace*(1+pos_term_uni))
            p_pos += (1-bigram_lambda)*np.log(p_posword_uni)

            if temp_word_uni in neg_uni:
                p_negword_uni = (neg_uni[temp_word_uni]+unigram_laplace)/(neg_totfreq_uni + unigram_laplace*(1+neg_term_uni))
            else:
                p_negword_uni = unigram_laplace/(neg_totfreq_uni + unigram_laplace*(1+neg_term_uni))
            p_neg += (1-bigram_lambda)*np.log(p_negword_uni)

        for j in range(len(dev_set[i])-1):
            temp_word = dev_set[i][j] + " " + dev_set[i][j+1]

            if temp_word in pos_bi:
                p_posword_bi = (pos_bi[temp_word]+bigram_laplace)/((pos_totfreq_uni + pos_totfreq_bi) + bigram_laplace*(1+(pos_term_uni + pos_term_bi)))
            else:
                p_posword_bi = bigram_laplace/((pos_totfreq_uni + pos_totfreq_bi) + bigram_laplace*(1+(pos_term_uni + pos_term_bi)))

            p_pos += bigram_lambda*np.log(p_posword_bi)

            if temp_word in neg_bi:
                p_negword_bi = (neg_bi[temp_word]+bigram_laplace)/((neg_totfreq_uni + neg_totfreq_bi) + bigram_laplace*(1+(neg_term_uni + neg_term_bi)))
            else:
                p_negword_bi = bigram_laplace/((neg_totfreq_uni + neg_totfreq_bi) + bigram_laplace*(1+(neg_term_uni + neg_term_bi)))

            p_neg += bigram_lambda*np.log(p_negword_bi)

        if (p_pos >= p_neg) :
            result.append(1)
        else:
            result.append(0)

    print(result)

    return result
