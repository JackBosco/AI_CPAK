"""
	Regression analysis for the alignment data using linear regression
	Author: Jack Bosco
"""
import config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read in the data, split into X and y
df = pd.read_csv(config.treated_path, index_col=0)
X = df.iloc[:, [0, 1, 6, 7, 8]] # preop hka, preop jlo, age, bmi, femoral transverse rotation
y = df.iloc[:, 2] # planned hka

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) # 90% training, 10% testing, 42 is the answer to everything

# standardize the data
scaler = StandardScaler()
X_train_scalar = scaler.fit_transform(X_train)
X_test_scalar = scaler.transform(X_test)

# linear regression model
lin = LinearRegression()

# fit the model with polynomial features
poly = PolynomialFeatures(degree=4) # 5 degrees because 5 features
X_train_poly = poly.fit_transform(X_train_scalar)
X_test_poly = poly.transform(X_test_scalar)
poly.fit(X_train_poly, y_train)
lin.fit(X_train_poly, y_train)

# make some predictions
y_pred = lin.predict(X_test_poly)
error = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for linear regression: {error}")
y_pred_train = lin.predict(X_train_poly)
error_train = mean_squared_error(y_train, y_pred_train)
print(f"Mean Squared Error for linear regression on training data: {error_train}")


fig = plt.figure()

plt.scatter(X.iloc[:, 0], y)
plt.plot(X_train.iloc[:, 0].sort_values(), pd.Series(y_pred_train).sort_values(), color='red')
plt.xlabel('Pre-op HKA')
plt.ylabel('Predicted Planned HKA')
plt.show()