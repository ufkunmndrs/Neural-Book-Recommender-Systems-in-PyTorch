#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 14:07:39 2020

@author: Ufkun-Bayram Menderes

This Python module contains the necessary functions to encode the goodbooks10k
dataset for preprocessing
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def proc_col(col, train_col=None):
    """
    Encodes pandas Dataframe with continuous, corresponding ids

    Parameters
    ----------
    col : pandas.core.frame.DataFrame
        Pandas dataframe whose id's will be continously encoded
    train_col : pandas.core.framee.Dataframe, optional
        Training dataframe. The default is None.

    Returns
    -------
    pandas.core.frame.DataFrame
        Pandas Dataframe with continously encoded id's

    """
    if train_col is not None:
        unique = train_col.unique()
    else:
        unique = col.unique()
    name2idx = {k:v for v,k in enumerate(unique)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(unique)



def encode_data(datfram, train=None):
    """
    Encodes ratings.csv file into continuous user and book id's respectively

    Parameters
    ----------
    datfram : pandas.core.frame.DataFrame
        pandas dataframe to be ordered continuously with corresponding columns
    train : pandas.core.frame.DataFrame
        additional dataframe to be ordered

    Returns
    -------
    datfram : pandas.core.frame.DataFrame
        ratings dataframe with continuously ordered user_id and book_id 
        columns

    """
    datfram = datfram.copy()
    for col_name in ['user_id', 'book_id']:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(datfram[col_name], train_col)
        datfram[col_name] = col
        datfram = datfram[datfram[col_name] >= 0]
    return datfram