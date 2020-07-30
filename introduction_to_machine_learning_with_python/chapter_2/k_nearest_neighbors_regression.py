#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn
import pandas as pd


# In[3]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)

# fit the model using the train data and training tragets
reg.fit(X_train, y_train)


# In[4]:


# Now we can make prediction on the test set:
print("Test set predictions: \n{}".format(reg.predict(X_test)))


# In[6]:


# We can also evaluate the model using the score method
# which for regressor returns the R^2 score (the coefficient of determination).
# Measure of the goodness of a prediction for a regression model
# Yields a score between 0 and 1. 
# 1 corresponds to perfect match
# 0 corresponds to a constant model that just predicts the mean of the training set responses
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


# In[9]:


# ==========Analyzing KNeighborsRegressor===========================
fig, axes = plt.subplots(1,3, figsize=(15,4))

#create 1000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9], axes):
    # make predictions using 1, 3 or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, '^', c=mglearn.cm2(1), markersize=8)
    
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test))
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")


# In[ ]:




