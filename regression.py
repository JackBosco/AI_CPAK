"""
	Regression analysis for the alignment data using linear regression
	Author: Jack Bosco
"""
from curses import beep
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
import pickle

import numpy as np

# read in the data, split into X and y
df = pd.read_csv(config.treated_path, index_col=0)
X = df.iloc[:, [0]] #1, 6, 7, 8]] # preop hka, preop jlo, age, bmi, femoral transverse rotation
y = df.iloc[:, 2] # planned hka

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 75% training, 25% testing, 42 is the answer to everything

# standardize the data
scaler = StandardScaler()
X_train_scalar = scaler.fit_transform(X_train)
X_test_scalar = scaler.transform(X_test)

# normalize the data with min-max scaling
normalizer = MinMaxScaler(feature_range=(-1, 1))
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

def test_model(fit_model, model_name, testset, trainset, three_d=False):
	y_pred = fit_model.predict(testset[0])
	error = mean_squared_error(testset[1], y_pred)
	print(f"Mean Squared Error for {model_name}: {error}")
	y_pred_train = fit_model.predict(trainset[0])
	error_train = mean_squared_error(trainset[1], y_pred_train)
	print(f"Mean Squared Error for {model_name} on training data: {error_train}")
	print(f"Score: {fit_model.score(testset[0], testset[1])}")

	y_all = np.concatenate((y_pred, y_pred_train))
	x_app = np.concatenate((X_test.iloc[:,0], X_train.iloc[:, 0]))
	dt1 = pd.DataFrame({'0':np.array(x_app), '1':pd.Series(y_all)})
	dt = pd.DataFrame({'0':np.array(X_test.iloc[:, 0]),'1':pd.Series(y_pred)})
	dt2 = pd.DataFrame({'0':np.array(X_train.iloc[:, 0]),'1':pd.Series(y_pred_train)})
	dt.sort_values(by='0', inplace=True)
	dt1.sort_values(by='0', inplace=True)
	dt2.sort_values(by='0', inplace=True)

	# plot the training data
	plt.scatter(X_train.iloc[:, 0], y_train, color='blue', label='Training Data')

	# plot the line of best fit
	plt.plot(dt1.loc[:, '0'], dt1.loc[:, '1'], color='red', label='Regression Line of Best Fit')
	
	# add axis lines
	plt.axvline(x=-2)
	plt.axvline(x=2)
	plt.axhline(y=-2)
	plt.axhline(y=2)

	# plot the error
	yerr = np.abs(dt.loc[:,'0']-dt.loc[:,'1'])
	plt.fill_between(dt['0'], dt['1']+yerr, dt['1']-yerr, color='red', alpha=.2, label='Error Room')
	plt.fill_between(dt['0'], (dt['1']+.33*yerr), (dt['1']-.33*yerr), color='red', alpha=.4, label='33% Error Room')
	
	plt.title(f'{model_name} Model Prediction for Planned aHKA from Pre-op aHKA')
	plt.xlabel('Pre-op aHKA')
	plt.ylabel('Planned aHKA')
	plt.legend()
	plt.show()
	plt.savefig(f'{model_name}'.replace(' ','_') +'_regression.png')
	plt.close()
	


def do_lin():
	# linear regression model
	lin = LinearRegression()
	# fit the model with polynomial features
	poly = PolynomialFeatures(degree=3) # sigmoidal or tanh or something
	X_train_poly = poly.fit_transform(X_train_scalar)
	X_test_poly = poly.transform(X_test_scalar)
	lin.fit(X_train_poly, y_train)

	test_model(lin, 
			"linear regression",
				testset=(X_test_poly, y_test),
				trainset=(X_train_poly, y_train))
	
	
# feed-forward neural network model
def do_mlp(train=True):
	if train:
		mlp = MLPRegressor()
		clf = GridSearchCV(mlp, {'hidden_layer_sizes': [(5,), (10,), (20,), (30,)],
								'max_iter': [1000, 2000, 3000],
								'solver': ['sgd', 'adam'],
								'activation': ['tanh']})
		clf.fit(X_train_normalized, y_train)
		best = clf.best_estimator_
		pickle.dump(best, open('neural_network.h5', 'wb'))
	else:
		best=pickle.load(open('neural_network.h5', 'rb'))
		
	test_model(best, 
			"neural network",
				testset=(X_test_normalized, y_test),
				trainset=(X_train_normalized, y_train))


# Support Vector Machine model
def do_svm(train=True):
	if train:
		parameters = {'kernel':('linear', 'rbf'),
					'degree': [2, 3, 4],
					'C':[1, .5, .1], 'gamma':[1, .1, .01, .001, .0001]}
		svr = NuSVR()
		clf = GridSearchCV(svr, parameters)
		clf.fit(X_train_scalar, y_train)
		best = clf.best_estimator_
		pickle.dump(best, open('support_vector_machine.h5', 'wb'))
	else:
		best=pickle.load(open('support_vector_machine.h5', 'rb'))
	
	test_model(best,
			"support vector machine",
				testset=(X_test_scalar, y_test),
				trainset=(X_train_scalar, y_train))

# Gaussian Process model
def do_gaus():
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
	kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
	gp.fit(X_train_scalar, y_train)
	test_model(gp,
			"gaussian process",
				testset=(X_test_scalar, y_test),
				trainset=(X_train_scalar, y_train))

if __name__ == '__main__':
	do_lin()
	do_mlp()#train=False) # this takes a long time to run (R.I.P. YOUR CPU)
	do_svm()#train=False)
	#do_gaus()