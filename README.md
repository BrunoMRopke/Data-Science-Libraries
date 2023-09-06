# Data-Science-Libraries
This repository has the intention to provide practical examples for data science tasks using python.


Data Science Libraries
1. Data Manipulation and Analysis
Pandas
Description: A powerful library for data manipulation and analysis, offering data structures for efficiently storing large datasets and tools to wrangle and analyze them.
Example:
python
Copy code
import pandas as pd
# Creating a DataFrame from a dictionary
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# Displaying the first rows of the DataFrame
print(df.head())
2. Numerical Computation
NumPy
Description: Fundamental package for numerical computations in Python. It provides support for arrays (including matrices) and many mathematical functions to operate on these arrays.
Example:
python
Copy code
import numpy as np
# Creating a 1D array
arr = np.array([1, 2, 3, 4, 5])
# Calculating the mean of the array values
print(np.mean(arr))
3. Data Visualization
Matplotlib

Description: A plotting library for creating static, animated, and interactive visualizations in Python.
Example:
python
Copy code
import matplotlib.pyplot as plt
# Plotting a simple line graph
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
Seaborn

Description: Built on top of Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics.
Example:
python
Copy code
import seaborn as sns
# Setting the theme for the plot
sns.set_theme()
# Loading a built-in dataset
tips = sns.load_dataset("tips")
# Creating a relational plot
sns.relplot(x="total_bill", y="tip", data=tips);
4. Machine Learning
Scikit-learn
Description: Provides simple and efficient tools for predictive data analysis. It features various classification, regression, and clustering algorithms.
Example:
python
Copy code
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# Loading the iris dataset
X, y = load_iris(return_X_y=True)
# Training a logistic regression model
clf = LogisticRegression().fit(X, y)
5. Deep Learning
TensorFlow

Description: An open-source framework for machine learning and deep learning, allowing developers to create large-scale neural networks with many layers.
Example:
python
Copy code
import tensorflow as tf
# Loading the mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
PyTorch

Description: An open-source machine learning library, used for a range of tasks including deep learning.
Example:
python
Copy code
import torch
# Creating a random tensor
x = torch.rand(5, 3)
print(x)
