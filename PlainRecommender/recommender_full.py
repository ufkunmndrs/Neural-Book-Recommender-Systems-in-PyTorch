#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:01:39 2020
Python module containing the full classic Recommender
The code is mainly built on Rob Zacharski's "The Ancient Art of the Numerati"
and contains all the modules and functions implemented in it:
http://guidetodatamining.com/
The class was further expanded in order to introduce more functionality and 
more methods for recommendation

@author: Ufkun-Bayram Menderes
"""
from math import sqrt
import codecs
import pandas as pd

# These were just some data and functions for me to test out their 
# correct implementation
# users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
#                       "Norah Jones": 4.5, "Phoenix": 5.0,
#                       "Slightly Stoopid": 1.5,
#                       "The Strokes": 2.5, "Vampire Weekend": 2.0},
         
#          "Bill":{"Blues Traveler": 2.0, "Broken Bells": 3.5,
#                  "Deadmau5": 4.0, "Phoenix": 2.0,
#                  "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
         
#          "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
#                   "Deadmau5": 1.0, "Norah Jones": 3.0, "Phoenix": 5,
#                   "Slightly Stoopid": 1.0},
         
#          "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
#                  "Deadmau5": 4.5, "Phoenix": 3.0,
#                  "Slightly Stoopid": 4.5, "The Strokes": 4.0,
#                  "Vampire Weekend": 2.0},
         
#          "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
#                     "Norah Jones": 4.0, "The Strokes": 4.0,
#                     "Vampire Weekend": 1.0},
         
#          "Jordyn":  {"Broken Bells": 4.5, "Deadmau5": 4.0,
#                      "Norah Jones": 5.0, "Phoenix": 5.0,
#                      "Slightly Stoopid": 4.5, "The Strokes": 4.0,
#                      "Vampire Weekend": 4.0},
         
#          "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
#                  "Norah Jones": 3.0, "Phoenix": 5.0,
#                  "Slightly Stoopid": 4.0, "The Strokes": 5.0},
         
#          "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
#                       "Phoenix": 4.0, "Slightly Stoopid": 2.5,
#                       "The Strokes": 3.0}
#         }




# def manhattan_distance(rating1:dict, rating2:dict) -> float:
#     if rating1 == None or rating2 == None:
#         raise ValueError("No proper User as input")
#     difference = 0
#     for key in rating1: #iterate over every entry
#         if key in rating2: #compare only entries which are listed in both dicts
#             difference += abs(rating1[key] - rating2[key])
#     return difference

# def nearest_neighbor(username, users):
#     distances = []
#     for user in users: #for every user in our dict
#         if user != username: #Handle own username as exception
#             distance = minkowski_distance(users[user], users[username], 2)
#             distances.append((distance, user))
#     distances.sort()
#     return distances

# def max_distance(username,users):
#     return max(nearest_neighbor(username, users))

# def min_distance(username, users):
#     return min(nearest_neighbor(username, users))

# def total_distances(username, users):
#     return sum(nearest_neighbor(username, users))

# def recommend(username, users):
#     nearest = nearest_neighbor(username, users)[0][1] #return username of nearest neighbor
#     recommendations = []
#     neighbor_ratings = users[nearest] #extract rated bands of nearest user
#     user_ratings = users[username]
#     for artist in neighbor_ratings:
#         if artist not in user_ratings:
#             recommendations.append((artist, neighbor_ratings[artist]))
#     return sorted(recommendations, key=lambda artist_tuple: artist_tuple[1],
#                   reverse = True)

# def minkowski_distance(rating1, rating2, p):
#     distance = 0
#     common_ratings = False
#     for key in rating1:
#         if key in rating2:
#             distance += pow(abs(rating1[key] - rating2[key]), p)
#             common_ratings = True
#     if common_ratings == True:
#         return pow(distance, 1/p)
#     else:
#         return 0
            
# def pearson(rating1, rating2):
#     sum_xy = 0
#     sum_x = 0
#     sum_y = 0
#     sum_x2 = 0
#     sum_y2 = 0
#     n = 0
#     for key in rating1:
#         if key in rating2:
#             n += 1 #increment n if both users rated same entity
#             x = rating1[key]
#             y = rating2[key]
#             sum_xy += x * y
#             sum_x += x
#             sum_y += y
#             sum_x2 += x**2
#             sum_y2 += y**2
#     if n == 0:
#         return 0
#     #Nenner berechnen 
#     denominator = sqrt(sum_x2 - (sum_x**2) / n) * sqrt(sum_y2 - (sum_y**2) / n)
#     if denominator == 0:
#         return 0
#     else:
#         return (sum_xy - (sum_x*sum_y)/n) / denominator
    
    
# def nearest_pearson_neighbor(username, users):
#     global correlations
#     correlations = []
#     for user in users:
#         if user != username:
#             single_cor = pearson(users[username], users[user])
#             correlations.append((single_cor, user))
#     correlations.sort()
#     return correlations

            
# def top_nearest(username, users):
#     return nearest_pearson_neighbor(username, users)[-1]
    
# def dot_product(rating1, rating2):
#     for key in rating1:
#         if key in rating2:
#             dot = sum(rating1[key]*rating2.get(key, 0) for key in rating1)
#     return dot

# def cosine_sim(rating1, rating2):
#     counter = dot_product(rating1, rating2)
#     sum_r1 = []
#     sum_r2 = []
#     for vals in rating1.values():
#         sum_r1.append(pow(vals, 2))
#     for vals in rating2.values():
#         sum_r2.append(pow(vals, 2))
#     denom = sqrt(sum(sum_r1)) * sqrt(sum(sum_r2))
#     return counter / denom

# def nearest_cosine_neighbor(username, users):
#     neighbors = []
#     for user in users:
#         if user != username:
#             cosine_similarity = cosine_sim(users[username], users[user])
#             neighbors.append((cosine_similarity, user))
#     neighbors.sort()
#     return neighbors

# def nearest_cos_neighbor(username, users):
#     return nearest_cosine_neighbor(username, users)[-1]

# def avg_distance(rating1, rating2):
#     common_rated = []
#     distance = minkowski_distance(rating1, rating2, 2)
#     for key in rating1:
#         if key in rating2:
#             common_rated.append(key)
#     return distance / len(common_rated)

# def avg_distance_total(username, users):
#     total_distance = 0
#     for user in users:
#         if user != username:
#            total_distance += avg_distance(users[username], users[user])
#     return total_distance

# def nearest_avg_neighbor(username, users):
#     neighbors = []
#     for user in users:
#         if user != username:
#             average_distances = avg_distance(users[username], users[user])
#             neighbors.append((average_distances, user))
#     neighbors.sort()
#     return neighbors

class recommender():
    def __init__(self, data, k=1, metric='pearson', n=5):
        """
        Recommender class which takes user input data and implements various
        functions for both collaborative filtering and item-based filtering

        Parameters
        ----------
        data : dict
            is the dictionary in which our data, i.e. the user ratings, are contained
        k : int
            determines k number of neighbors we are considering for knn calculation,
            default is k=1
        metric : function
            determines the user similarity metric which we are using for 
            collaborative filtering
            The default is 'pearson' metric, other option is cos_similarity
        n : int, optional
            number of top ratings, the default is 5.

        Returns
        -------
        None.

        """
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        self.frequencies = {}
        self.deviations = {}
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        if self.metric == 'cos_similarity':
            self.fn = self.cos_similarity
        if type(data).__name__ == 'dict':
            self.data = data
            
    def convertProductID2name(self, id):
        """
        Function which converts given product id's into names

        Parameters
        ----------
        id : product id, the key in the dictionary
            id of the respective products

        Returns
        -------
        self.productid_to_name[id], id
            if the product is in the dictionary, return its value
        else simply return id

        """
        #if the key id is in the dictionary
        if id in self.productid2name:
            #then return its value, which is the product name
            return self.productid2name[id]
        #else simply return the id
        else:
            return id
    
    def user_ratings(self, id, n):
        """
        Function which returns n-top items rated by a user

        Parameters
        ----------
        id : user id
            id of users for who we will return ratings for
        n : int
            number of top ratings which will be returned 

        Returns
        -------
        n top ratings

        """
        # user id is the key in our dictionary, its values are the ratings of 
        # the respective users
        # the ratings as values are in themselves again a dictionary with products
        # as keys and their ratings as values
        print("Ratings for: " + self.userid2name[id])
        #s ave those ratings for ever user with an id 
        ratings = self.data[id]
        # print the length of the values
        print(len(ratings))
        # make a list out of key-value pairs in ratings as tuples
        ratings = list(ratings.items())
        # call the upper function, pass the keys of the tuples as function 
        # arguments and make a list comprehension for key-value pairs in the 
        # tuples
        ratings = [(self.convertProductID2name(k), v) for (k, v) in 
                   ratings]
        # sort the list according to the first element of each pair, which is 
        # the rating, reverse the list so we have top values at the beginning
        ratings.sort(key = lambda artist_tuple: artist_tuple[1], reverse=True)
        # slice the list up till index n so only n number of elements will be returned
        ratings = ratings[:n]
        for rating in ratings:
            print("%s\t%i" % (rating[0], rating[1]))
            
    def load_book_db(self, path=''):
        """
        Function which specifically loads the BX books database/dataset

        Parameters
        ----------
        path : TYPE, optional
            DESCRIPTION. The default is ''.
            Function that loads our book database
            Path is where BX book dataset is located
        Each for Loop is for one of the csv-files in our BX-Dump file
        1st for-loop: loads book ratings made by users
        2nd for-loop: loads book information
        3rd for-loop: loads user information

        Returns
        -------
        None.

        """
        # initialize data dict
        self.data = {}
        # initialize counter
        i = 0
        # load book ratings into self.data in read mode with utf-8 encoding
        f = codecs.open(path + "BX-Book-Ratings.csv", 'r', 'utf-8')
        # iterate over each line in the file
        # lines are structured as: user-id, isbn, user ratings
        for line in f:
            i += 1 # increment counter
            # separate each line into fields by splitting at semicolon
            fields = line.split(";")
            # extract user-id
            user = fields[0].strip('"')
            # extract isbn
            book = fields[1].strip('"')
            # extract user rating
            rating = int(fields[2].strip().strip('"'))
            
            # if the uer in the dictionary
            if user in self.data:
                # then return his current ratings
                current_ratings = self.data[user]
            # if not
            else:
               # create empty ratings dictionary
               current_ratings = {}
            # inser ratings as values for book keys in ratings dictionary
            current_ratings[book] = rating
            # assign dict current_ratings as value to key user
            self.data[user] = current_ratings
        f.close()
        # Now load books into self.productid2name
        # Books contains isbn, title, and author among other fields
        f = pd.read_csv(path + 'books.csv')
        for line in f:
            i += 1 #increment counter
            fields = line.split(";")
            isbn = fields[0].strip('"')
            title = fields[1].strip('"')
            author = fields[2].strip('"')
            title = title + ' by ' + author
            # assign title as value to key isbn
            self.productid2name[isbn] = title
        f.close()
        #
        #  Now load user info into both self.userid2name and
        #  self.username2id
        #
        f = codecs.open(path + 'BX-Users.csv', 'r', 'utf-8')
        for line in f:
            i += 1 #increment counter
            fields = line.split(';')
            user_id = fields[0].strip('"')
            location = fields[1].strip('"')
            # if age field is filled with an entry, hence why fields is > 3
            if len(fields) > 3:
                age = fields[2].strip().strip('"')
            # if not, than enter NULL into age field
            else:
                age = 'NULL'
            # if an entry for age field apparent, enter 
            if age != 'NULL':
                # then concatetenate info into one string
                value = location + '  (age: ' + age + ')'
            else:
                value = location
            # assign value as value for key user id in dict
            self.userid2name[user_id] = value
            # assign user_id as value to key location
            self.username2id[location] = user_id
        f.close()
        print(i)
    
    def load_goodbooks(self):
        """
        Function to load goobooks 10k ratings and books.csv file

        Returns
        -------
        creates self.data dictionary

        """
        self.data = {}
        # initialize counter
        i = 0
        # ratings
        f = pd.read_csv("../goodbooks10kDataset/ratings.csv")
        # books
        b = pd.read_csv("../goodbooks10kDataset/books.csv")
        ratings_list = list(zip(f.user_id, f.book_id, f.rating))
        books_list = list(zip(b.id, b.title, b.authors))
        for index, tup in enumerate(ratings_list):
            i += 1
            user = tup[0]
            book = tup[1]
            rating = tup[2]                    #
            if user in self.data:
                # then return his current ratings
                current_ratings = self.data[user]
            # if not
            else:
               # create empty ratings dictionary
               current_ratings = {}
            # inser ratings as values for book keys in ratings dictionary
            current_ratings[book] = rating
            # assign dict current_ratings as value to key user
            self.data[user] = current_ratings
        k = 0
        for index, book_tup in enumerate(books_list):
            k += 1 #increment counter
            book_id = book_tup[0]
            title = book_tup[1]
            author = book_tup[2]
            title = title + ' by ' + author
            # assign title as value to key isbn
            self.productid2name[book_id] = title
        print(i)
        
        
    def compute_deviations(self):
        """
        Computes item deviations in a given dataset 
        
        args: self, no other argument necessary
        
        Functions computes deviations for each item according to the given 
        formula in Chapter 3 of "The ancient art of Numerati"
        
        After function is called, self.deviations will be filled with each item 
        as key, and its values are dictionaries in which keys represent other items 
        and values represent the deviation to that item in float type

        Returns
        -------
        None.

        """
        # for each person in the data :
        #get their ratings
        for ratings in self.data.values():
            #for each item and its rating in tthe ratings dict
            for (item, rating) in ratings.items():
                self.frequencies.setdefault(item, {})
                self.deviations.setdefault(item, {})
                # for each item2 & rating2 in that set of ratings: 
                for(item2, rating2) in ratings.items():
                    # add the difference between the ratings to our computation
                    if item != item2:
                        self.frequencies[item].setdefault(item2, 0)
                        self.deviations[item].setdefault(item2, 0.0)
                        self.frequencies[item][item2] += 1
                        self.deviations[item][item2] += rating - rating2
        for (item, ratings) in self.deviations.items():
            for item2 in ratings:
                ratings[item2] /= self.frequencies[item][item2]
    
        
    def slope_one_recommend(self, user):
        """
        Implementation of slope one algorithm for recommendation and 
        prediction of an item for a given user, for item-based recommendation

        Parameters
        ----------
        user : str
            user for which we will recommend items and predict his ratings
            self.compute_deviations function must be called before calling this 
            function
            

        Returns
        -------
        recommendations : list
            A list containing tuples as elements
            tuple[0] = item which user hasn't rated yet, type: str
            tuple[1] = predicted rating for that unrated item, type: float

        """
        self.compute_deviations()
        recommendations = {}
        frequencies = {}
        # for every item and rating in the user's recommendations
        for (user_item, user_rating) in self.data[user].items():
            # for every item in our dataset that the user didn't rate
            # example. dict_items([('Taylor Swift', 5), ('PSY', 2)])
            for (diff_item, diff_ratings) in self.deviations.items():
                if diff_item not in self.data[user] and \
                   user_item in self.deviations[diff_item]:
                # get the deviations, frequencies for the item that wasn't rated
                # but only if a value for deviations exists and diff_item is not 
                # in user_ratings
                    freq = self.frequencies[diff_item][user_item]
                    # get the frequency of diff_item and user_item, 
                    # key in frequencies is the different item, second dict indice 
                    # is the frequency with user_item
                    recommendations.setdefault(diff_item, 0.0)
                    frequencies.setdefault(diff_item, 0)
                    # add to the running sum representing the numerator
                    # of the formula 
                    # diff_item is key, result of formula is value
                    recommendations[diff_item] += (diff_ratings[user_item] +
                                                   user_rating)* freq
                    # keep a running sum of the frequency of diffitem
                    frequencies[diff_item] += freq
        recommendations = [(self.convertProductID2name(k), v / frequencies[k])
                           for (k, v) in recommendations.items()]
        recommendations.sort(key=lambda artist_tuple: artist_tuple[1], reverse = True)
        return recommendations
    
    
    
    def minkowski_distance(self, rating1, rating2, p):
        """
        Implementation of Minkowski distance formula

        Parameters
        ----------
        rating1 : dict
            dictionary in which the ratings of user1 are specified, 
            passing as argument into function via "users['users']"
        rating2 : dict
            dictionary in which ratings of user2 are specified, same 
            argument passing conventions as above
        p : int
            specifies parameter 'p' in minkowski distance formula
            p = 1 --> Manhattan distance
            p = 2 --> Euclidean distance

        Returns
        -------
        float
            returns the minkowski distance measure between rating1 and rating 2
            according to parameter 'p'

        """
        distance = 0
        common_ratings = False
        for key in rating1:
            if key in rating2:
                distance += pow(abs(rating1[key] - rating2[key]), p)
                common_ratings = True
        if common_ratings == True:
            return pow(distance, 1/p)
        else:
            return 0
            
        
        
    def pearson(self, rating1, rating2):
        """
        Implementation of Pearson correlation coefficient formula
        Calculates similarity of users via ratings ina given dataset

        Parameters
        ----------
        rating1 : dict
            ratings of the first user 
        rating2 : dict
            ratings of 2nd user
        function implements pearson similarity metric by comparing commonly rated 
        elements of both users

        Returns
        -------
        float
            returns a float which measures the similarity of two user according 
            to their set of commonly given ratings, ranges from -1 to 1

        """
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # compute denominator
        denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n)
                       * sqrt(sum_y2 - pow(sum_y, 2) / n))
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator
    
    def dot(self, rating1, rating2):
        """
        Computes the dot product SPECIFICALLY for our application
        at hand and without using numpy data structures

        Parameters
        ----------
        rating1 : dict
            dict containing user ratings for user 1, must be indiced in the form 
            of 'users[user]' as function argument in order to properly access user 
            ratings
        rating2 : dict
            dict containing user ratings for user 2, same as above for indicing

        Returns
        -------
        dot : float
            returns the dot product for 2 users 

        """
        for key in rating1:
            if key in rating2:
                dot = sum(rating1[key]*rating2.get(key, 0) for key in rating1)
        return dot
    
    def cos_similarity(self, rating1, rating2):
        """
        Implementation of cosine similarity w.r.t to given datastructures at hand

        Parameters
        ----------
        rating1 : dict 
            user ratings for user 1, dict must be indiced and indexing must be 
            passed as function argument in form of (users['user'])
        rating2 : dict
            user ratings for user 2, same as above 

        Returns
        -------
        cosine similarity
            alternative measure for capturing the similarity between two users, 
            ranges from 0 to 1

        """
        num = self.dot(rating1, rating2)
        sum_r1 = []
        sum_r2 = []
        for vals in rating1.values():
            sum_r1.append(pow(vals, 2))
        for vals in rating2.values():
            sum_r2.append(pow(vals, 2))
            denom = sqrt(sum(sum_r1)) * sqrt(sum(sum_r2))
        return num / denom
        
        
    def averages(self, users):
        """
        Computes average ratings assigned by users in a given dataset

        Parameters
        ----------
        users : dict
            Dictionary in which users as keys and their respective ratings for 
            the respective items (dict) as values are stored

        Returns
        -------
        results : dict
            users are keys, their average ratings (float) are values

        """
        results = {}
        for (key, ratings) in users.items():
            results[key] = float(sum(ratings.values())) / len(ratings.values())
        return results
    
    
    
    def item_similarity(self, artist1, artist2, user_ratings):
        """
        Computes the similarity between two items in a given dataset

        Parameters
        ----------
        artist1 : str
            artist1 for which we will compute the similarity with artist2, must be 
            in user_ratings dict
        artist2 : str
            artist1 for which we will compute the similarity with artist2, must be 
            in user_ratings dict    
        user_ratings : dict
            dictionary containing users as keys, their ratings for items 
            as internal dict as values

        Returns
        -------
        float
            similarity between two items

        """
        
        averages = {}
        for (key, ratings) in user_ratings.items():
            #take the values of the internal dict, sum over each value and divide by length of values
            averages[key] = (float(sum(ratings.values()))
                      / len(ratings.values()))
        numerator = 0
        denom1 = 0
        denom2 = 0
        for (user, ratings) in user_ratings.items():
            #if the users rated both artists
            if artist1 in ratings and artist2 in ratings:
                #compute the averages of each user
                avg = averages[user]
                # in ratings dict, artists are the keys and their respective values, 
                # which we access via indexing, are their ratings
                # those ratings for the artists are subtracted by the average of each user
                # and then multiplied
                numerator += (ratings[artist1] - avg) * (ratings[artist2] - avg)
                denom1 += (ratings[artist1]-avg)**2 
                denom2 += (ratings[artist2]-avg)**2
        return numerator / (sqrt(denom1) * sqrt(denom2))
    
    def normalized_rating(self, user:str, item:str):
        """
        Normalizes the rating of an item for a user in a given dataset according 
        to minimum and maximum rating of that dataset

        Parameters
        ----------
        user : str
            a user who has rated the item function argument
        item : str
            item that a user has rated, is within the internal dictionary as key
        

        Returns
        -------
        float
            normalized rating for a user given said item

        """
        global min_rat
        global max_rat
        min_ratings = []
        max_ratings = []
        for (key, rating) in self.data.items():
            key_max = max(rating.keys(), key=(lambda k: rating[k]))
            key_min = min(rating.keys(), key=(lambda k: rating[k]))
            min_ratings.append(rating[key_min])
            max_ratings.append(rating[key_max])
        min_rat = min(min_ratings)
        max_rat = max(max_ratings)
        if item in self.data[user]:
            return (2*((self.data[user][item]-min_rat)) - (max_rat-min_rat)) / (max_rat-min_rat)
            
    
    def denormalized_rating(self, user:str, item:str):
        """
        Returns a normalized rating back into a scaled rating, 
        given respective rating scale
        
        BEWARE!!! This function takes an user and an item and then gives the denormalized
        rating for it, the denormalizer only takes a float value

        Parameters
        ----------
        user : str
            DESCRIPTION.
        norm_rating : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return (0.5*((self.normalized_rating(user, item)+1) * (max_rat-min_rat))) + min_rat
        
    
    def denormalize(self, norm_rating:float):
        """
        

        Parameters
        ----------
        norm_rating : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return (0.5*((norm_rating+1) * (max_rat-min_rat))) + min_rat

        
    def predict_rating(self, user:str, item:str):
        """
        Predicts the rating of an item from a user from a given dataset

        Parameters
        ----------
        user : str
            user in a given dataset
        item : str
            an item which the user HAS NOT rated yet

        Returns
        -------
        tuple
            tuple[0] = user 
            tuple[1] = item recommended for that user
            tuple[2] = predicted rating for that item

        """
        #initiate numerator and denumerator of formula
        denom = 0
        numerator = 0
        for rated_item in self.data[user]:
            if rated_item in self.data[user]:
                denom += (self.item_similarity(rated_item, item, self.data)
                          *self.normalized_rating(user, rated_item))
                numerator += abs(self.item_similarity(item, rated_item, self.data))
        return (user, item, self.denormalize(denom/numerator))
   
        
        
            
    def nearest_neighbor(self, username):
        """
        Determines nearest neighbors, i.e. users for a user in a given dataset
        for collaborative filtering.
        Default metric is Pearson correlation coefficient, can be changed to other
        metrics when instatiating an object of Recommender class 

        Parameters
        ----------
        username : str
            username of a user in the set of our users
        
        since self.fn == 'pearson', the nearest neighbors will be calculated 
        using pearson coefficient, can be changed if desired

        Returns
        -------
        distances : list
            a list containing all the distances from the user to the other users
            in the total set of users
            Elements are tuples with users (str) as first element and the distances
            to them as floats as second element

        """
        distances = []
        for instance in self.data:
            # measure distance only if users are distinct 
            if instance != username:
                # Pearson metric is applied
                distance = self.fn(self.data[username], self.data[instance])
                distances.append((instance, distance))
        distances.sort(key=lambda artist_tuple: artist_tuple[1], reverse=True)
        return distances
    
    def recommend(self, user:str):
        """
        Recommends an item from the dataset to the user

        Parameters
        ----------
        user : str
            user in the set of users

        Returns
        -------
        type = list
            returns a list of top n recommendations for the user
            list[0] == item recommended, type str
            list[1] == predicted rating, type float

        """
        # create empty dictionary for recommendations
        recommendations = {}
        # compute the nearest neighbors of a specific user
        nearest = self.nearest_neighbor(user)
        # extract ratings for this specific user from self.data dict
        user_ratings = self.data[user]
        # initialize total distance counter
        total_distance = 0.0
        # iterate over k elements, here k=1
        for i in range(self.k):
            total_distance += nearest[i][1]
        # add up the total distance by summing over the ratings of k nearest neighbors
        # compute the contribution of each k neighbor
        for i in range(self.k):
            weight = nearest[i][1] / total_distance
            # extract name of each of the k neighbors
            name = nearest[i][0]
            # extract ratings of each person out of the k neighbors
            neighbor_ratings = self.data[name]
        # for every artist in neighbor ratings
        for artist in neighbor_ratings:
            # if the artist is not in the user_ratings
            if not artist in user_ratings:
                # and if the artist is not already in recommendations dict 
                # i.e. artists that the neigbor rated but the user didn't
                # for key artists, assign the value of its neighbor rating times the weight 
                # it contributes
                if artist not in recommendations:
                    # for key artists, assign the value of its neighbor rating times the weight 
                    # it contributes
                    recommendations[artist] = (neighbor_ratings[artist]*weight)
                else:
                    # if artist already existing, assign neighbor rating + recommendations rating
                    # and multiply it with weight
                    recommendations[artist] = (recommendations[artist]+neighbor_ratings[artist] * weight)
        # make a list out of the key, value pairs of artist and recommendations tuples
        recommendations = list(recommendations.items())
        # convert the product id to names via list comprehension
        recommendations = [(self.convertProductID2name(k), v) for (k, v) in recommendations]
        # sort the list by rating, reverse in order for top ratings to be at the beginning
        recommendations.sort(key=lambda artist_tuple: artist_tuple[1], reverse=True)
        # return n elements
        return recommendations[:self.n]
    
    def nearest_items(self, user:str):
        """
        Recommends an item to a user by recommending the most similar item 
        to highest rated items by a user

        Parameters
        ----------
        user : str
            user for which we will the items most similar to his top rated item
            

        Returns
        -------
        tuple
        tuple[1] = user, str
        tuple[2] = recommendations, list       

        """
        # compute deviations since self.deviations will be useful for 
        # future calculatios
        self.compute_deviations()
        # sort ratings of a user
        top_rated = {item: rating for item, rating in 
                     sorted(self.data[user].items(), key=lambda value: value[1],
                            reverse=True)}
        # discard all elements which are not top rated
        for key, rating in top_rated.items():
            # determine highest rated item(s) and their rating(s)
            max_user_rating = max(top_rated.keys(), key=(lambda k: top_rated[k]))
            max_user_rating = top_rated.get(max_user_rating)
            #finding elements with multiple values
            mult_top_rated = [k for k, v in top_rated.items() if v == max_user_rating]
        # now we find the most similar item for each top rated item of a user
        similarities = []
        for item in mult_top_rated:
            for unrated_item in self.deviations:
                if unrated_item not in top_rated:
                    item_sim = self.item_similarity(item, unrated_item, self.data)
                    similarities.append((item_sim, item, unrated_item))
        item_near = {}
        # we create a dictionary in which we compute the similarities of every
        # top rated item to other top rated items, where top rated items are keys
        # and list of similarity to an item and item itself as tuples are elements
        # of that list
        for sim, item, near_item in similarities:
            if item in item_near:
                item_near[item].append((sim, near_item))
            else:
                item_near[item] = [(sim, near_item)]
        # sort the values according to their similarity, highest similarity 
        # at top
        for key, rating in item_near.items():
            rating.sort(key=lambda sim: sim[0], reverse=True)
        recommendations = [item_near[item][0] for item in item_near]
        # remove duplicate items, remove the duplicate with lower item similarity
        for sim, item in recommendations:
            for sim2, item2 in recommendations:
                if item == item2 and sim < sim2:
                    recommendations.remove((sim, item))
                elif item == item2 and sim > sim2:
                    recommendations.remove((sim2, item2))
        final_recoms = [elems[1] for elems in recommendations]
        return user, (final_recoms)
                    
    


if __name__ == '__main__':
    users2 =  {"David": {"Imagine Dragons": 3, "Daft Punk": 5,
     "Lorde": 4, "Fall Out Boy": 1},
     "Matt": {"Imagine Dragons": 3, "Daft Punk": 4,
     "Lorde": 4, "Fall Out Boy": 1},
     "Ben": {"Kacey Musgraves": 4, "Imagine Dragons": 3,
     "Lorde": 3, "Fall Out Boy": 1},
     "Chris": {"Kacey Musgraves": 4, "Imagine Dragons": 4,
     "Daft Punk": 4, "Lorde": 3, "Fall Out Boy": 1},
     "Tori": {"Kacey Musgraves": 5, "Imagine Dragons": 4,
     "Daft Punk": 5, "Fall Out Boy": 3}}
    
    #this is just a dummy in order to load data so that the recommender object
    #can be created, lone 1029 will load the actual data
    recommender = recommender(users2)
    recommender.load_goodbooks()
    
    



    
        
        
        
    
         
                
            
            
            
            
    
            
            
        
        
        
        
        
        
                            
                
                
                
            
                    
                    
                           
            
                           
            
                
                    
                    
                
                
                
                
                
                
                
            
                

            
            
        







