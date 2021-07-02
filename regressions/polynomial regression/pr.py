import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
print("independant data:")
print(x)
print("dependant data:")
print(y)

# Training the linear regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Training the polynomial regression model on the whole dataset
poly_reg = PolynomialFeatures(degree = 10)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualising the linear regression result
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x),color='blue')
plt.title('truth')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visualising the polynomial regression result
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly),color='blue')
plt.title('truth')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Predicting a new result with linear regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
