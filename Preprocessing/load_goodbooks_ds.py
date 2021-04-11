#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 03:04:35 2020

@author: Ufkun-Bayram Menderes

This Python Module contains the functions to load, partly preprocess and prepare
the goodbooks10k dataset and its files ratings.csv, books.csv, to_read.csv
and tags.csv
It also contains the preparation of the goodbooks10k dataset into training, validation
and testing, which will be imported by the 3 corresponding modules

"""
import pandas as pd
from encode_books import proc_col, encode_data
import numpy as np


###############################################################################
# 1st part: This file will provide necessary function for loading the goodbooks
# 10k dataset, in particular the "ratings.csv" file of the dataset which contains
# item_id, user_id, rating triplets
###############################################################################

def load_ratings():
    """
    Loads the 'ratings.csv' file of goodbooks 10k ratings dataset

    Returns
    -------
    ratings : pandas.core.frame.DataFrame
        Pandas dataframe
        1st column = book_id,
        2nd column = user_id,
        3rd colums = rating (1-5 scale)

    """
    ratings = pd.read_csv('../goodbooks10kDataset/ratings.csv', sep=',', error_bad_lines=False, encoding='latin-1')
    ratings = ratings[['user_id', 'book_id', 'rating']]
    return ratings

def load_books():
    """
    Loads the 'books.csv' file of goodbooks 10k dataset, contains useful metadata
    about books

    Returns
    -------
    books : pandas.core.frame.DataFrame
        books dataframe containing metadata about books in goodbooks dataset

    """
    books = pd.read_csv('../goodbooks10kDataset/books.csv', sep=',', 
                        error_bad_lines=False, encoding='latin-1')
    # 23 columns is lots of information - let's drop some of it
    # some columns might be helpful in forming embeddings later on
    books.drop(['image_url', 'small_image_url', 'isbn13', 'work_text_reviews_count',
                'best_book_id', 'work_id', 'language_code', 'ratings_count',
                'work_ratings_count', 'ratings_1', 'ratings_2',
                'ratings_3', 'ratings_4', 'ratings_5', 'books_count', 'original_title', 
                'book_id'], axis=1, inplace=True)
    books.columns = ['id', 'isbn', 'authors', 'year', 'title', 'avg_rating']
    return books

def load_to_read():
    """
    Loads 'to_read.csv' file of goodbooks dataset
    Can be used for specific recommendations later on in order to recommend items
    that user tagged as 'to read' AND would give a positive rating

    Returns
    -------
    to_read : pandas.core.frame.DataFrame
        Dataframe containing user id and book id (book to be read)
        1st column: user_id
        2nd column: book_id

    """
    to_read = pd.read_csv('../goodbooks10kDataset/to_read.csv', sep=',', error_bad_lines=False, encoding='latin-1')
    return to_read

def load_tags():
    """
    Loads 'tags.csv' file of goodbooks dataset
    contains tag_id for books and tag_name for name of the tag 

    Returns
    -------
    tags : pandas.core.frame.Dataframe
        1st column: tag_id's as ints
        2nd column: tag_names as str's'

    """
    tags = pd.read_csv('../goodbooks10kDataset/tags.csv', sep=',', error_bad_lines=False, encoding='latin-1')
    return tags

def list_2_dict(id_list:list):
    """
    

    Parameters
    ----------
    id_list : list
        DESCRIPTION.

    Returns
    -------
    d : dict
        DESCRIPTION.

    """
    d={}
    for id, index in zip(id_list, range(len(id_list))):
        d[id] = index
    return d

################################################################################
# This part of the file contains the preparation of training, validation and 
# testset. Training set makes up 80% of the entire dataset, with validation 
# and test making up 10% each. The variables will be imported by the corresponding
# modules so that each module gets the exact same data to train and test with
##############################################################################

# load ratings.csv into ratings variable
ratings = load_ratings()
# load books.csv into books variable
books = load_books()
# set number for random sapling of floats
np.random.seed(3)
#random sampling of floats
msk = np.random.rand(len(ratings)) < 0.8
#training set
train = ratings[msk].copy()
#validation set
valid = ratings[~msk].copy()
# split validation set into two parts
test_valid_split = np.array_split(valid, 2)
valid = test_valid_split[0]
# test set
test = test_valid_split[1]

num_users_total = ratings.user_id.nunique()
num_items_total = ratings.user_id.nunique()    
# encode training, validation and test data
train_ratings = encode_data(train)
valid_ratings = encode_data(valid, train)
test_ratings = encode_data(test)
