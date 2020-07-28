import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("Keys of iris_data: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + '\n...')

# The value of the key target_names is an array of strings, comtaining the species
# of flower that we want to predict
print("Target name: {}".format(iris_dataset["target_names"]))

# feature_names : list of strings giving description of each feature
print("Feature name: \n{}".format(iris_dataset['feature_names']))

print("Type of data: {}".format(type(iris_dataset['data'])))

print("Shape of data: {}".format(iris_dataset['data'].shape))

print("First five rows of data: \n{}".format(iris_dataset['data'][:5]))

print("Type of target: {}".format(type(iris_dataset['target'])))

print("Shape of target: {}".format(iris_dataset['target'].shape))

print("Target: \n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_train shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_test.shape))

# Create dataframe from data in X_train
# label the columns using strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataFrame, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 25),
      marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# plt.show()


# Machine Learning Model (K-Nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making prediction
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# Evaluating the model

y_pred = knn.predict(X_test)
print("Test set predictions: {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))