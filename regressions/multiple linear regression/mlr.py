import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = np.array(dataset.iloc[:, :-1])
y = np.array(dataset.iloc[:,-1])
print("independant data:")
print(x)
print("dependant data:")
print(y)

#encoding the independant variables
ct = ColumnTransformer([('encoder',OneHotEncoder(),[-1])], remainder='passthrough')
x = np.array(ct.fit_transform(x),dtype = np.str)
print("encoding the independant variables")
print(x)

# Splitting dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print("x(independant variables) training:")
print(x_train)
print("x testing:")
print(x_test)
print("y(dependant variables) training:")
print(y_train)
print("y testing:")
print(y_test)

# Training the multiple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test  set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
