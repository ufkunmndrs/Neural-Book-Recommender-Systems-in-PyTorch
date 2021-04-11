#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:58:08 2020

@author: Ufkun-Bayram Menderes

This Python Module contains the implementation of Recommender Systems for the 
goodbooks10k dataset based on Biased Matrix Factorization.
Training for model is provided via train.py modul
The Model is implemented in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from load_goodbooks_ds import *
from encode_books import proc_col, encode_data
from train import train_epochs, test_loss
import pandas as pd
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


class Bias_MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        """
        Bias Matrix factorization class which includes includes 
        biases both in the instantiation of an object of this class 
        and the forward pass

        Parameters
        ----------
        num_users : int
            number of users in a given dataset/pandas dataframe
        num_items : int
            number of items in a given dataset/pandas dataframe
        emb_size : int, optional
            embedding size. The default is 100.

        Bias MF object
        -------
        None.

        """
        super(Bias_MF, self).__init__()
        # user embeddings
        self.user_embedding = nn.Embedding(num_users, emb_size)
        # bias embeddings for user
        self.user_bias = nn.Embedding(num_users, 1)
        # item embeddings
        self.item_embedding = nn.Embedding(num_items, emb_size)
        # bias embeddings for item
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_embedding.weight.data.uniform_(0,0.05)
        self.item_embedding.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        
    def forward(self, user, item):
        """
        Forward pass method of the Bias_MF class
        includes biases for both users and items into the computation

        Parameters
        ----------
        user : int
            user and its numerical representation as user_ids
            
        item : int
            items (in this case books) and their numerical representations
            as book_ids

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        user_vector = self.user_embedding(user)
        item_vector = self.item_embedding(item)
        bias_user = self.user_bias(user).squeeze()
        bias_item = self.item_bias(item).squeeze()
        return (user_vector*item_vector).sum(1) +  bias_user  + bias_item
    
    def recommend(self, user_id, n=150, r=3):
        """
        Recommends n top items to a user 

        Parameters
        ----------
        user_id : int
            integer representation of a user, will be converted to a torch.Tensor
        n : int, optional
            number of top recommendations which we will return.
            The default is 10.
        
        r : int, optional
            Decimal places to which predicted rating shall be rounded up to

        Returns
        -------
        recommendations : list
            list top n recommendations with predicted ratings

        """
        if user_id > max_user_id or user_id < min_user_id:
            raise ValueError("Invalid user ID")
        books = load_books()
        user = torch.tensor([user_id])
        books_recommend = torch.tensor(ratings.book_id.unique().tolist()) - 1
        predictions = model(user, books_recommend).tolist()
        # normalize the ratings since many ratings are over 5.0
        predictions = [i/max(predictions)*max_rating for i in predictions]
        # zip the book_ids and corresponding rating prediction to a list of 
        # (rating, id) tuples
        book_ids = ratings.book_id.unique()
        predictions = list(zip(predictions, book_ids))
        # get the books which the user has alread read
        user_books_read = ratings[ratings.user_id == user]
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
        
        


if __name__ == 'main':
    model = Bias_MF(num_users, num_items, emb_size=100)
    train_epochs(model, epochs=4, lr=0.01)
    
    
    test_users = torch.LongTensor(test_ratings.user_id.values)
    test_books = torch.LongTensor(test_ratings.book_id.values)
    test_predictions = model(test_users, test_books)
    
    # Normalize test predictions
    test_predictions = test_predictions.tolist()
    test_predictions = [i/max(test_predictions)*max_rating for i in test_predictions]
    test_predictions = list(np.around(test_predictions, 0))


