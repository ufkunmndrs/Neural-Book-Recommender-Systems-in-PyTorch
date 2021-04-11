#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 05 01:22:46 2020

@author: Ufkun-Bayram Menderes

This Python Module contains the implementation of 
Neural Recommender Class and its training on the goodbooks10k dataset. The Neural
Recommender is implemented in PyTorch.
"""
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import the loaded test, validation and trainingdata
from load_goodbooks_ds import *
from train import train_epochs, test_loss
from encode_books import proc_col, encode_data
from metrics import *




###############################################################################
# 1st part: Loading the goodbooks dataset from the load_goodbooks_ds.py file
# In that file, the dataset is alredy split up into training, validation and test
# sets. 
# train_ratings = Training set, valid_ratings = validation set, test_rating = 
# testset. The training set consists of 80% of the data in the goodbooks10k 
# dataset, while validation and training make up 10% each. 
###############################################################################


# number of users for embeddings = 53424
num_users = train_ratings.user_id.nunique()
# number of books = 10000
num_items = train_ratings.book_id.nunique()
# determine min and max ratings
min_rating = min(ratings["rating"])
max_rating = max(ratings["rating"])
second_highest = max_rating - 1

min_user_id = min(ratings["user_id"])
max_user_id = max(ratings["user_id"])

print("Number of users: {}, Number of books: {}, Min rating: {}, Max rating: {}"
      .format(num_users_total, num_items_total, min_rating, max_rating))

###############################################################################
# Now we create the RecommenderNet Class which is our Neural Recommender based 
# on a Neural Network with one hidden layer, reLu activation 
# for hidden layers, and default dropout of 0.01
###############################################################################


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        """
        PyTorch Module for Neural Recommender on goodbooks10k dataset

        Parameters
        ----------
        num_users : int
            number of unique users in the dataset
        num_items : int
            number of unique items(books) in the dataset
        emb_size : int, optional
            size of user-item embeddings. The default is 100.
        n_hidden : int, optional
            (output) size of hidden layers. The default is 10.
        

        Returns
        -------
        RecommenderNet object
        RecommenderNet object as Neural Recommender

        """
        super(RecommenderNet, self).__init__()
        # initialize embedding layers
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        # initialize hidden layers and linear transformation
        self.hl1 = nn.Linear(emb_size*2, n_hidden)
        self.hl2 = nn.Linear(n_hidden, 1)
        # set dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, user, item):
        """
        Forward pass for the Neural Recommender

        Parameters
        ----------
        user : int
            user_id, numerical representation of a user in the goodbooks dataset
        item : int
            book_id, numerical representation of a user 

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        """
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        x = F.relu(torch.cat([user_vector, item_vector], dim=1))
        x = self.dropout(x)
        x = F.relu(self.hl1(x))
        x = self.hl2(x)
        return x
    
    def recommend(self, userID, n=150, r=3):
        """
        Computes the recommendations for a user with collaborative filtering

        Parameters
        ----------
        userID : int
            user ID in the goodbooks dataset, must be be between 1 and
            53424
        n : int, optional
             number of recommendations returned by the system.
             The default is 10.
        
        r : int, optinal
            number of decimal places to which the predicted ratings are rounded 
            up to. The default is 3

        Returns
        -------
        recommended_books : pandas
            pandas dataframe containing user recommendation with predicted rating
            columns in order: id, 

        """
        # Verify that user is correct
        if userID > max_user_id or userID < min_user_id:
            raise ValueError("Invalid user ID")
        #load preprocessed books.csv file from goodbooks
        books = load_books()
        # create book vector for books
        books_vec = torch.tensor(ratings['book_id'].unique().tolist()) - 1
        # create user vector
        user_vec = torch.tensor([userID])
        # get shape of book vector
        books_shape = list(books_vec.shape)
        books_shape = books_shape[0]
        # resize user vector to a compatible size with book vector
        user_vec = torch.cat([user_vec]*books_shape)
        # compute ratings prediction for user by forwardpass
        predictions = self.forward(user_vec, books_vec)
        # make numpy array out of predictions
        predictions = np.array([a[0] for a in predictions])
        # get book id's into a list
        book_ids = ratings.book_id.unique()
        # convert predictions into a list with float elements
        predictions = predictions.tolist()
        # get predicted item ratings out of tensor elements
        # make a list out of the ratings
        predictions = [i.item() for i in predictions]
        # zip the book_ids and corresponding rating prediction to a list of 
        # (rating, id) tuples
        predictions = list(zip(predictions, book_ids))
        # get the books which the user has alread read
        user_books_read = ratings[ratings.user_id == userID]
        # make a list out of the book_id's that user has alread read
        read_list = list(user_books_read.book_id)
        # make an empty list for unread books, fill it with book_ids
        # of books that user has not rated/read yet
        unread_books = []
        for book in predictions:
            if book[1] not in read_list:
                unread_books.append(book)
        # sort the elements according to their rating, top ratings at top
        unread_books = sorted(unread_books, key=lambda book_tuple: book_tuple[0],
                              reverse=True)
        # indice the list on desired top n elements
        top_n_books = unread_books[:n]
        # get book_ids of top n elements
        top_n_book_ids = [books[1] for books in top_n_books]
        # get the full book information of these books from books.csv dataset
        recommended_books = books[books["id"].isin(top_n_book_ids)]
        # make dict out of top_n_books, keys=ratings, values=ids
        book_rating_dict = dict(top_n_books)
        # reverse keys and values s.t. ids are keys, ratings are values
        book_rating_dict = {value:key for key, value in book_rating_dict.items()}
        # make tuples out key, value elements, save them in the new list
        books_list_rev = [(k, v) for k, v in book_rating_dict.items()]
        # add a new dataframe with id, predicted rating tuples
        book_rating_tups = pd.DataFrame(books_list_rev, columns=['id','predicted_rating'])
        # merge both dataframes into one, id's column is common columns
        recommended_books = recommended_books.merge(book_rating_tups)
        # change predicted ratings >= 5.0 to 5
        recommended_books.loc[recommended_books['predicted_rating'] > 5.0, 'predicted_rating'] = 5.0
        # sort the predicted_rating column such that ratings with highest values
        # are at the top with descending order
        recommended_books = recommended_books.sort_values('predicted_rating', ascending=False)
        # round the values for predicted ratings
        recommended_books = recommended_books.round({'predicted_rating':r})
        # create new columns for predicted ratings on a 1-5 scale
        recommended_books['predicted_rating_scaled'] = recommended_books['predicted_rating']
        # round the ratings to zero decimal places so only values from 1-5 exist
        recommended_books = recommended_books.round({'predicted_rating_scaled':0})
        # return recommended books
        return recommended_books
        
        
if __name__ == '__main__':
    # initialize the model by creating object of RecommenderNet class   
    model = RecommenderNet(num_users, num_items, emb_size=100)
    # train the model on test and validation dataset with weight decay
    train_epochs(model, epochs=2, lr=0.05, wd=1e-6, unsqueeze=True)
    
    
    # create test inputs and calculate test predictions
    test_users = torch.LongTensor(test_ratings.user_id.values)
    test_books = torch.LongTensor(test_ratings.book_id.values)
    # test predictions are returned as torch.tensor object
    test_predictions = model(test_users, test_books)
    # make list out of test_predictions
    test_predictions = test_predictions.tolist()
    # round the predictions zero decimal places 
    test_predictions = [ratings[0] for ratings in test_predictions]
    test_predictions = list(np.around(test_predictions, 0))

