#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn
import pandas as pd


# In[ ]:


# =============Liner Regression on wave dataset======================


# In[2]:


mglearn.plots.plot_linear_regression_wave()


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
[]
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

print("Training set score: {}".format(lr.score(X_train, y_train)))
print("Test set score: {}".format(lr.score(X_test, y_test)))


# In[6]:


# =============Liner Regression on Cancer dataset======================


# In[7]:


X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)


print("Training set score: {}".format(lr.score(X_train, y_train)))
print("Test set score: {}".format(lr.score(X_test, y_test)))


# In[ ]:




