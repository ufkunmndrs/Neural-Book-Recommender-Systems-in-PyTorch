#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 01:55:14 2020

PyTorch module containing the necessary metrics for evaluating the Recommender 
Systems
Metrics included are:
@author: Ufkun-Bayram Menderes
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

ratings = load_ratings()
min_rating = min(ratings["rating"])
max_rating = max(ratings["rating"])
second_highest = max_rating - 1

def regression_loss(test_predictions):
    """
    Calculates the regression loss on the test data for the ratings

    Parameters
    ----------
    test_predictions : list
        List containing all the predictions for the test dataset
        

    Returns
    -------
    reg_loss : float
        Regression loss; tells us how much the recommender was off on 
        average for the rating predictions on the test dat

    """
    true_ratings = list(test_ratings.rating)
    total_loss = list(np.array(true_ratings) - np.array(test_predictions))
    total_loss = [abs(diff) for diff in total_loss]
    reg_loss = sum(total_loss)/len(total_loss)
    return reg_loss

def classification_loss(test_predictions, diff=1):
    """
    Calculates the classfication loss of the models on the test dataset
    Also includes calculation of accuracy for ratings

    Parameters
    ----------
    test_predictions : list
        List containing predicted ratings for test dataset
        
    diff : int, optional
        The difference between actual rating and predicted 
        rating for each rating in the test dataset. The default is 1.

    Raises
    ------
    ValueError
        If user enters a difference that is invalid, i.e. exceeds or is below
        difference of max rating and min rating for a given dataset

    Returns
    -------
    loss_count : int
        absolute number of incorrectly predicted ratings
    accuracy : float
        accuracy of the recommender on the test data.
    off_by_diff_count : int
        absolute number of predicted ratings that were off by diff
    accuracy_per : float
        absolute predicted ratings that were off by diff among the 
        entire test data
    off_by_diff_percentage : float
        percentage of predicted ratings that were off by diff among incorrectly
        predicted ratings
    """
    if diff > second_highest or diff < min_rating:
        raise ValueError("Invalid Difference")
    true_ratings = list(test_ratings.rating)
    truth_pred_pairs = list(zip(true_ratings, test_predictions))
    loss_count = 0
    for i in truth_pred_pairs:
        if i[0] != i[1]:
            loss_count += 1
    accuracy = loss_count / len(true_ratings) * 100
    accuracy = 100 - round(accuracy, 3)
    off_by_diff_count = 0
    off_by_diff_percentage = 0
    for i in truth_pred_pairs:
        if abs(i[0] - i[1]) == diff:
            off_by_diff_count += 1
            off_by_diff_percentage = off_by_diff_count / loss_count
    # computes accuracy of each rating
    accuracy_per = off_by_diff_count / len(test_ratings.rating)
    return loss_count, accuracy, off_by_diff_count, accuracy_per, off_by_diff_percentage

def recall_per(test_predictions, rating):
    """
    Calculates the recall for a rating on a given scale

    Parameters
    ----------
    test_predictions : list
        list containing all the predicted ratings for a test dataset
    rating : int
        Rating on a scale for which we will calculate the 
        recall on a testdataset
        
    Raises
    ------
    ValueError
        If user enters a difference that is invalid, i.e. exceeds or is below
        the min rating for a given dataset

    Returns
    -------
    rating recall : float
        Recall for the selected rating in percent

    """
    if rating > max_rating or rating < min_rating:
        raise ValueError("Invalid rating")
    true_ratings = list(test_ratings.rating)
    truth_pred_pairs = list(zip(true_ratings, test_predictions))
    correct_count_total = 0
    for items in truth_pred_pairs:
        # True positives
        if(items[0] == rating and items[1] == rating):
                correct_count_total += 1
    # counting False Negatives
    truepos_falseneg = list(true_ratings).count(rating)
    rating_recall = (correct_count_total / truepos_falseneg)       
    return rating_recall




def precision_per(test_predictions, rating):
    """
    Calculates the precision on the test data for a rating on a given scale

    Parameters
    ----------
    test_predictions : list
        list containing the predicted ratings for the test data 
    rating : int
        rating for which the predicision on the test data will
        be returned
    
    Raises
    ------
    ValueError
        If user enters a difference that is invalid, i.e. exceeds or is below
        the min rating for a given dataset
        
    Returns
    -------
    rating_precision : float
        Precision for selected rating in per cent

    """
    if rating > max_rating or rating < min_rating:
        raise ValueError("Invalid rating")
    true_ratings = list(test_ratings.rating)
    truth_pred_pairs = list(zip(true_ratings, test_predictions))
    ratings_list = list(range(1, max_rating+1))
    correct_count_total = 0
    for items in truth_pred_pairs:
        #countint True Positives
        if(items[0] == rating and items[1] == rating):
                correct_count_total += 1
    # counting All Positives
    pos_total = test_predictions.count(rating)
    rating_precision = 0
    try:
        rating_precision = (correct_count_total / pos_total)
    except ZeroDivisionError:
        print("Rating {} wasn't predicted".format(rating))
    return rating_precision 

def recall_macro_avg(test_predictions):
    """
    Calculates the macro averaged recall of the Recommender on the test 
    dataset

    Parameters
    ----------
    test_predictions : list
        Contains all the predicted ratings on the test dataset

    Returns
    -------
    recalls/len(ratings_list) : float
        the sum of the recalls averaged over the dataset (in per cent)

    """
    recalls = 0
    ratings_list = list(range(min_rating, max_rating+1))
    for r in ratings_list:
        recalls += recall_per(test_predictions, r)
    return recalls/len(ratings_list)

def precision_macro_avg(test_predictions):
    """
    Calculates the macro averaged precision over the test dataset

    Parameters
    ----------
    test_predictions : list
        Python list containing all the predicted ratings on the testdataset

    Returns
    -------
    precisions/len(ratings_list) : float
        The sum of all the precisions for each rating averaged over the dataset
        (in per cent)
        

    """
    precisions = 0
    ratings_list = list(range(min_rating, max_rating+1))
    for r in ratings_list:
        precisions += precision_per(test_predictions, r)
    return precisions/len(ratings_list)


# Warning! Code contains bug and must be refactored first before using
def f_measure_per(test_predictions, rating):
    """
    Calculates the f-measure for a rating value in the test dataset

    Parameters
    ----------
    test_predictions : list 
        contains all the predicted ratings on the test dataset
    rating : int
        rating from the scale of the test dataset for which we will measure the
        recall
    
    Raises
    ------
    ValueError
        If user enters a difference that is invalid, i.e. exceeds or is below 
        min rating of the dataset


    Returns
    -------
    float
        f-measure for the rating

    """
    if rating > max_rating or rating < min_rating:
        raise ValueError("Invalid rating")
    numerator = precision_per(test_predictions, rating) * recall_per(test_predictions, rating)
    denom = precision_per(test_predictions, rating) + recall_per(test_predictions, rating)
    return (2*(numerator/denom))

# Warning! Code contains bug and must be refactored before proper usage!
def f_measure_macro_avg(test_predictions):
    """
    calculates the macro averaged f-measure for the predictions of the 
    neural recommender

    Parameters
    ----------
    test_predictions : list
        contains all the predicted ratings for the test dataset

    Returns
    -------
    f_measures : float
        macro averaged f-measure for the test dataset

    """
    f_measures = 0
    ratings_list = list(range(min_rating, max_rating+1))
    for r in ratings_list:
        f_measures += f_measures_per(test_predictions, r)
    return f_measures

def accuracy_per(test_predictions, rating):
    """
    calculates the accuracy for a rating on the rating scale from the 
    test dataset

    Parameters
    ----------
    test_predictions : list
        contains all the predicted ratings for the test dataset
    rating : int
        rating for which the accuracy will be returned
    
    Raises
    ------
    ValueError
        If user enters a difference that is invalid, i.e. exceeds or is below
        min rating for a given dataset

    Returns
    -------
    rating_accuracy : TYPE
        DESCRIPTION.

    """
    if rating > max_rating or rating < min_rating:
        raise ValueError("Invalid rating")
    accuracy = 0
    true_ratings = list(test_ratings.rating)
    truth_pred_pairs = list(zip(true_ratings, test_predictions))
    correct_count_total = 0
    for items in truth_pred_pairs:
        # True positives
        if(items[0] == rating and items[1] == rating):
                correct_count_total += 1
    # All positives
    pos_total = test_predictions.count(rating)
    true_rating_count = true_ratings.count(rating)
    # true negatives by subtracting the desired true label 
    # from truth labels list
    true_neg = len(true_ratings) - true_rating_count
    numerator = correct_count_total + true_neg
    denom = len(true_ratings)
    rating_accuracy = numerator/denom
    return rating_accuracy



