import os
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 900)

import queue
import threading
import time
import multiprocessing as mp
from multiprocessing import Lock, Process, Queue, current_process
import queue
from functools import partial
import numpy as np


def calc_support_of_column(item1, item2):
    
    res = list(map(partial(calc_support, item2), item1))
    return pd.Series(res, index=item1)


def calc_support(item1, item2, relative=True):
    conjunction = 0
    joint = 0

    for transaction in transactions:
        if item1 in transaction and item2 in transaction:
            conjunction += 1
        if item1 in transaction or item2 in transaction: 
            joint += 1
    if relative:
        return conjunction/joint
    else:
        return conjunction/len(transactions)


# Loading Data
data_path = os.path.join(os.getcwd(), 'data')
ratings = pd.read_csv(os.path.join(data_path, 'ratings_small.csv'),  low_memory=False)


# Create a Pivot Table with 1 if the user has rated the movie and 0 if he has not done so

# reduced the dataset to the needed features and convert it to a matrix
user_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_ratings = user_ratings.astype(bool).astype(int)


# Change every 1 to the according Movie ID (Column)
for idx, row in user_ratings.iterrows():
    for i in user_ratings.columns:
        if row[i] != 0:
            user_ratings.at[idx, i] = i
    if(idx % 100 == 0):
        print('idx: ', idx)
        

# Creating the itemsets
transactions = user_ratings.values.tolist()
for i in range(0, len(transactions)):
    transactions[i] = [value for value in transactions[i]if value != 0]
    
    
movie_items = set()
for list_ in transactions:
    movie_items.update(list_)

movie_items = sorted(movie_items)

support = pd.DataFrame(columns=movie_items, index=movie_items)
support = support[:10]


for i in range(len(support.columns)):
    print('column: ', i)
    support[support.columns[i]] = calc_support_of_column(support[support.columns[i]].index ,support.columns[i])


print(support.head())

support.to_csv('support_small_vectorized.csv')

    
