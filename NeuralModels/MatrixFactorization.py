#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 03 04:06:36 2020

@author: Ufkun-Bayram Menderes

This Python Module contains the implementation of Recommender Systems for the 
goodbooks10k dataset based on Matrix Factorization.
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

###############################################################################
# In this part, we will create two Matrix Factorization models with PyTorch 
# Embeddings.
# The first model will have no biases, the 2nd will have user and item biases 
# respectively, which will be added to the final factorization
###############################################################################


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        """
        Matrix Factorization class which embeds both users and items 
        into nn.Embeddings according to chosen embedding size

        Parameters
        ----------
        num_users : int
            number of users of a given dataset
        num_items : int
            number of items of a given dataset
        emb_size : int, optional
            size of each emebedding. The default is 100.

        Returns
        -------
        MF object

        """
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)
        self.user_embedding.weight.data.uniform_(0, 0.05)
        self.item_embedding.weight.data.uniform_(0, 0.05)
    
    def forward(self, user, item):
        """
        Performs computations of Matrix Factorization for users and books
        

        Parameters
        ----------
        user_vec : nn.Embedding
            Embedding layer of users and user_id's
        item_vec : nn.Embedding
            Embedding layer of items and item_id's

        Returns
        -------
        Output for Matrix Factorization

        """
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        return (user_vec*item_vec).sum(1)
    
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
        predictions = [i/max(predictions)*ratings.rating.max() for i in predictions]
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

# determining number of users and items for passing them as parameters
# for the object
num_users = train_ratings.user_id.nunique()
num_items = train_ratings.book_id.nunique()



if __name__ == '__main__':
    #train_epochs(MF_model, epochs=35, lr=0.005)
    model = MF(num_users, num_items, emb_size=100)
    train_epochs(model, epochs=100, lr=0.01)
    
    
    test_users = torch.LongTensor(test_ratings.user_id.values)
    test_books = torch.LongTensor(test_ratings.book_id.values)
    test_predictions = model(test_users, test_books)
    
    # normalize the ratings
    test_predictions = test_predictions.tolist()
    test_predictions = [i/max(test_predictions)*max_rating for i in test_predictions]
    test_predictions = list(np.around(test_predictions, 0))




