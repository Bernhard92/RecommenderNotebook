#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 900)
from apyori import apriori

import queue
import threading
import time
from threading import Lock
lock = Lock()


# # Loading Data

# In[2]:


data_path = os.path.join(os.getcwd(), 'data')
ratings = pd.read_csv(os.path.join(data_path, 'ratings_small.csv'),  low_memory=False)
#user = pd.read_csv(os.path.join(data_path, 'users.dat'), sep='::', engine='python', names=['UserID','Gender','Age','Occupation','Zip-code'])
#movies = pd.read_csv(os.path.join(data_path, 'movies_preprocessed.csv'))


# ### Create a Pivot Table with 1 if the user has rated the movie and 0 if he has not done so

# In[3]:


# reduced the dataset to the needed features and convert it to a matrix
user_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_ratings = user_ratings.astype(bool).astype(int)


# ### Change every 1 to the according Movie ID (Column)

# In[4]:


for idx, row in user_ratings.iterrows():
    for i in user_ratings.columns:
        if row[i] != 0:
            user_ratings.at[idx, i] = i
    if(idx % 100 == 0):
        print('idx: ', idx)
        


# ### Creating the itemsets

# In[5]:


transactions = user_ratings.values.tolist()
for i in range(0, len(transactions)):
    transactions[i] = [value for value in transactions[i]if value != 0]
    
movie_items = set()
for list_ in transactions:
    movie_items.update(list_)

movie_items = sorted(movie_items)


# # Calculating Support

# In[6]:


#item matrix with support of to items
def calc_support_of_column(value, column, relative=False):
    for i in range(0, len(column.index)):        
        column[column.index[i]] = calc_support(value, column.index[i], relative) 
    return column


# In[7]:


def calc_support(item1, item2, relative):
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


# # Creating Threads

# In[8]:


exitFlag = 0
class myThread (threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
    
    def run(self):
        print ("Starting " + self.name, flush=True)
        process_data(self.name, self.q)
        print ("Exiting " + self.name, flush=True)


# In[9]:



def process_data(threadName, q):
    counter = 0
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            column_index = q.get()
            queueLock.release()
            result = calc_support_of_column(
                support.columns[column_index], 
                support[support.columns[column_index]].copy(), 
                relative=True)
            #with lock:
            support[support.columns[column_index]] = result
            if (counter % 100 == 0):
                print ("%s processing %s" % (threadName, column_index), flush=True)
            counter += 1
        else:
            queueLock.release()
            time.sleep(1)


# ## Main

# In[ ]:


support = pd.DataFrame(columns=movie_items, index=movie_items)
support = support[:30]
threadList = []
for i in range(1, 30):
    threadList.append(("Thread-"+str(i)))

queueLock = threading.Lock()
workQueue = queue.Queue()

threads = []
threadID = 1

# Create new threads
for tName in threadList:
    thread = myThread(threadID, tName, workQueue)
    thread.start()
    threads.append(thread)
    threadID += 1

# Fill the queue
queueLock.acquire()
print('Filling the queue...')
for i in range(0, len(support)):
    workQueue.put(i)
    print('put: ', i)
queueLock.release()
print('Finished filling the queue!')

# Wait for queue to empty
while not workQueue.empty():
    pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
    t.join()
print ("Exiting Main Thread")


# In[ ]:


support.to_csv('support_small.csv')
# In[ ]:




