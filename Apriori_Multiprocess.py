#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 900)

import queue
import threading
import time
import multiprocessing as mp
from multiprocessing import Lock, Process, Queue, current_process


# # Calculating Support
#item matrix with support of to items
def calc_support_of_column(value, column, transactions, relative=True):

    for i in range(0, len(column.index)):        
        column[column.index[i]] = calc_support(value, column.index[i], transactions, relative) 
    return column


def calc_support( item1, item2, transactions, relative):
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


def process_data(column_index, support, transactions):
    pid = os.getpid()

    result = calc_support_of_column(
        support.columns[column_index], 
        support[support.columns[column_index]].copy(), 
        transactions
    )

    if (column_index % 100 == 0):
        print ("%s processing %s" % (pid, column_index), flush=True)
    return result


def main():
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

    pool = mp.Pool(processes=mp.cpu_count())

    # print the output
    resultFrame = pd.DataFrame()
    for i in range(len(support)):
        column = pool.apply(process_data, [i,  support, transactions])
        resultFrame[column.name] = column


    print ("Exiting Main Process", flush=True)

    resultFrame = resultFrame.sort_index(axis=1)
    resultFrame.to_csv('support_small_process.csv')
            

if __name__=='__main__':
    main()
