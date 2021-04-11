#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 14:04:47 2020

@author: Ufkun-Bayram Menderes
This Python Module contains the necessary training functions for the various 
implementations of Recommenders based on PyTorch. The functions are applicable 
for all 3 Recommender System Models.
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from load_goodbooks_ds import load_ratings, load_books
from encode_books import proc_col, encode_data


ratings = load_ratings()
np.random.seed(3)
#random sampling of floats
msk = np.random.rand(len(ratings)) < 0.8
#training set
train = ratings[msk].copy()
#validation set
valid = ratings[~msk].copy()
train_ratings = encode_data(train)
valid_ratings = encode_data(valid, train)


def train_epochs(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    """
    Trains a an Object of nn.Module
    In this usecase, it trains the Neural Recommender

    Parameters
    ----------
    model : nn.Module
        Chosen Machine Learning model for matrix factorization
    epochs : int, optional
        Epochs/Iterations. The default is 10.
    lr : float, optional
        Learning rate for the model. The default is 0.01.
    wd : float, optional
        weight decay for model . The default is 0.0.
    unsqueeze : bool, optional
        PyTorch's squeeze/unsqueeze functionality. The default is False.

    Returns
    -------
    None.
    Prints out test loss via test loss function

    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for e in range(epochs):
        users = torch.LongTensor(train_ratings.user_id.values)
        books = torch.LongTensor(train_ratings.book_id.values)
        ratings = torch.FloatTensor(train_ratings.rating.values)
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        pred = model(users, books)
        loss = F.mse_loss(pred, ratings)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch:{}, Loss:{}".format(e,loss.item()))
    test_loss(model, unsqueeze)

def test_loss(model, unsqueeze=False):
    """
    Computes test loss of a given model

    Parameters
    ----------
    model : nn.Module
        chosen ML model for Matrix Factorization
    unsqueeze : bool, optional
        Pytorch's squeeze/unsqueeze functionality. The default is False.

    Returns
    -------
    prints out epochs and corresponding loss in that epoch

    """
    #alternatively:
    #torch.no_grad(), though not explicitly required
    model.eval()
    users = torch.LongTensor(valid_ratings.user_id.values)
    books = torch.LongTensor(valid_ratings.book_id.values)
    ratings = torch.FloatTensor(valid_ratings.rating.values)
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    pred = model(users, books)
    loss = F.mse_loss(pred, ratings)
    print("test loss %.3f " % loss.item())