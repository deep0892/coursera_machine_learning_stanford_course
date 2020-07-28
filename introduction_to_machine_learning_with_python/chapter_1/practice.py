import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

# numpy array
x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

# Create a 2D NumPy array with a diagonal of ones and zero everywhere else (identity matrix)
eye = np.eye(4)
print("NumPy array: \n{}".format(eye))

# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

# Create sparse matrix using COO format:
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation\n{}".format(eye_coo))

# Generate a sequence of number from -10 to 10 with 100 step in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")

# Create a simple dataset of people
data = {
  "Name": ["John", "Anna", "Peter", "Linda"],
  "Location": ["Newn York", "Paris", "Berlin", "London"],
  "Age": [24,13,53,33]
}
data_pandas = pd.DataFrame(data)

# IPython.display allows "pretty printing of dataframes"
display(data_pandas)

# Select all rows that have age column >30
display(data_pandas[data_pandas.Age > 30])
