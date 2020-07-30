#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn
import pandas as pd


# In[ ]:


# ==================k-Nearest-Neighbors on Forge dataset==================


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


# In[15]:


# Visualization of decision boundary
# for one, three and nine neighbors
fig, axes = plt.subplots(1, 3, figsize=(20,4))
for n_neighbors, ax in zip([1, 3, 9], axes):
#     the fit methjod returns the object self, so we can 
#     fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


# In[16]:


# ==================k-Nearest-Neighbors on Cancer dataset==================


# In[25]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record trainign set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

