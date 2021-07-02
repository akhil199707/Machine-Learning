import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)
print("independant data:")
print(x)
print("dependant data:")
print(y)

# Feature scaling
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print("x")
print(x)
print("y")
print(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y.ravel())

# Predecting a new result
print("prediction")
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))))

# Visualising the SVR results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)),color='blue')
plt.title('truth')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# Visualising the SVR results(greater resuluion)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color='red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))),color='blue')
plt.title('truth')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
