#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn
import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# In[6]:


clf.fit(X_train, y_train)


# In[7]:


print("Test set predictions: {}".format(clf.predict(X_test)))


# In[8]:


print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# In[ ]:




