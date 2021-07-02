import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

# importing dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]
print("independant data:")
print(x)
print("dependant data:")
print(y)

#takeing care of missing data in dataset
#1) ignore the missing  data works for large data and 1% missing data
#2) replace the missing value by avg of total of that coloum
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x.iloc[:,1:3])
x.iloc[:,1:3] = imputer.transform(x.iloc[:,1:3])
print("takeing care of missing data in dataset:")
print(x)

#encoding the independant variables
ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x),dtype = np.str)
print("encoding the independant variables")
print(x)

#encoding the dependant variables
le = LabelEncoder()
y = le.fit_transform(y)
print("encoding the dependant variables:")
print(y)

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

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print("x train")
print(x_train)
print("x test")
print(x_test)
