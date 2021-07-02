import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
print("independant data:")
print(x)
print("dependant data:")
print(y)

# Training the decision tree regression model on the whole dataset
regressor = DecisionTreeRegressor()
regressor.fit(x,y)

# Predecting a new result
print(regressor.predict([[6.5]]))

# Visualising the decision tree regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color='red')
plt.plot(x_grid, regressor.predict(x_grid),color='blue')
plt.title('truth')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
