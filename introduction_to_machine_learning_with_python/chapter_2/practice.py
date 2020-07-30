#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import mglearn
import pandas as pd


# In[39]:


# generate forge dataset
X,y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))


# In[40]:


mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[41]:


mglearn.plots.plot_knn_classification(n_neighbors=3)


# In[14]:


# wave dataset
X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")


# In[26]:


# Wisconsin Breast Cancer dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys: {}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n {}".format(
    {n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))
print("Feature name: \n{}".format(cancer.feature_names))
print("Description: {}".format(cancer.DESCR))


# In[33]:


# Boston Housing Dataset
from sklearn.datasets import load_boston
boston = load_boston()
print("Description: {}".format(boston.DESCR))
print("Data shape: {}".format(boston.data.shape))

# Feature engineering: combination of features 
# to create new features 
X,y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))


# In[ ]:




