"""
	Regression analysis for the alignment data using linear regression
	Author: Jack Bosco
"""
from sklearn.svm import NuSVR
import config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# read in the data, split into X and y
df = pd.read_csv(config.treated_path, index_col=0)
X = df.iloc[:, [0, 1, 6, 7, 8]] # preop hka, preop jlo, age, bmi, femoral transverse rotation
y = df.iloc[:, 2] # planned hka

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 40% training, 10% testing, 42 is the answer to everything

# standardize the data
scaler = StandardScaler()
X_train_scalar = scaler.fit_transform(X_train)
X_test_scalar = scaler.transform(X_test)

# normalize the data with min-max scaling
normalizer = MinMaxScaler(feature_range=(-1, 1))
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

def test_model(fit_model, model_name, testset, trainset):
	y_pred = fit_model.predict(testset[0])
	error = mean_squared_error(testset[1], y_pred)
	print(f"Mean Squared Error for {model_name}: {error}")
	y_pred_train = fit_model.predict(trainset[0])
	error_train = mean_squared_error(trainset[1], y_pred_train)
	print(f"Mean Squared Error for {model_name} on training data: {error_train}")
	print(f"Score: {fit_model.score(testset[0], testset[1])}")

	plt.scatter(X.iloc[:, 0], y, color='blue')
	plt.plot(X_train.iloc[:, 0].sort_values(), pd.Series(y_pred_train).sort_values(), color='red')
	plt.title(f'{model_name} Model Prediction for Post-op HKA from Pre-op HKA')
	plt.xlabel('Pre-op HKA')
	plt.ylabel('Predicted Planned HKA')
	plt.show()
	plt.close()


# linear regression model
lin = LinearRegression()
# fit the model with polynomial features
poly = PolynomialFeatures(degree=2) # parabolas are cool
X_train_poly = poly.fit_transform(X_train_scalar)
X_test_poly = poly.transform(X_test_scalar)
lin.fit(X_train_poly, y_train)

test_model(lin, 
		   "linear regression",
			testset=(X_test_poly, y_test),
			trainset=(X_train_poly, y_train))

	
# feed-forward neural network model
def do_mlp():
	mlp = MLPRegressor()
	clf = GridSearchCV(mlp, {'hidden_layer_sizes': [(5,), (10,), (20,), (30,)],
							'max_iter': [1000, 2000, 3000],
							'solver': ['sgd', 'adam']})
	clf.fit(X_train_normalized, y_train)
	test_model(clf.best_estimator_, 
			"neural network",
				testset=(X_test_normalized, y_test),
				trainset=(X_train_normalized, y_train))

#do_mlp() # this takes a long time to run (R.I.P. YOUR CPU)

# Support Vector Machine model
parameters = {'kernel':('linear', 'rbf'),
			  'degree': [2, 3, 4],
			  'C':[1, .5, .1], 'gamma':[1, .1, .01, .001, .0001]}

svr = NuSVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train_scalar, y_train)
# print(cross_validate(clf, scaler.transform(X), y, cv=5))
test_model(clf.best_estimator_,
		   "support vector machine",
			testset=(X_test_scalar, y_test),
			trainset=(X_train_scalar, y_train))