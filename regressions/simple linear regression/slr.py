import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]
print("independant data:")
print(x)
print("dependant data:")
print(y)

# Splitting dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 1)
print("x(independant variables) training:")
print(x_train)
print("x testing:")
print(x_test)
print("y(dependant variables) training:")
print(y_train)
print("y testing:")
print(y_test)

# Training the Simple Linear Regression model on training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(x_test)

# Visualising training set result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Exp(Training set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()

# Visualising testing set result
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Exp(Testing set)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()
