"""
	Regression analysis for the alignment data using linear regression
	Author: Jack Bosco
"""
import os
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
X = df.loc[:, ['Pre-op mpta', 'Pre-op ldfa']] #1, 6, 7, 8]] # preop hka, preop jlo, age, bmi, femoral transverse rotation
y = df.loc[:, ['Planned MPTA', 'Planned LDFA']] # planned hka

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 75% training, 25% testing, 42 is the answer to everything

# standardize the data
scaler = StandardScaler()
X_train_scalar = scaler.fit_transform(X_train)
X_test_scalar = scaler.transform(X_test)
X_scalar = scaler.transform(X)

# normalize the data with min-max scaling
normalizer = MinMaxScaler(feature_range=(-1, 1))
out_normalizer = MinMaxScaler(feature_range=(-1, 1))

X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)
X_normalized = normalizer.transform(X)
y_train_norm = out_normalizer.fit_transform(y_train)
y_test_norm = out_normalizer.transform(y_test)
y_norm = out_normalizer.transform(y)

pickle.dump(normalizer, open(config.norm_path, 'wb'))
pickle.dump(out_normalizer, open(config.de_norm_path, 'wb'))


def test_model(fit_model, model_name, testset, trainset, x_data, norm1=out_normalizer):

	y_pred = fit_model.predict(testset[0])
	y_pred = pd.DataFrame(norm1.inverse_transform(y_pred))
	y_pred_train = fit_model.predict(trainset[0])
	y_pred_train = pd.DataFrame(norm1.inverse_transform(y_pred_train))
	y_pred_all = fit_model.predict(x_data)
	y_pred_all = norm1.inverse_transform(y_pred_all)
	y_test1, y_train1 = pd.DataFrame(y_test.iloc[:, 0] - y_test.iloc[:, 1]), pd.DataFrame(y_train.iloc[:, 0] - y_train.iloc[:, 1])

	# turn the data from MPTA and LDFA to aHKA
	X_test1 = X_test.iloc[:, 0] - X_test.iloc[:, 1]
	X_train1 = X_train.iloc[:, 0] - X_train.iloc[:, 1]
	X1 = X.iloc[:, 0] - X.iloc[:, 1]
	y_pred1 = y_pred.iloc[:, 0] - y_pred.iloc[:, 1]
	y_pred_train1 = y_pred_train.iloc[:, 0] - y_pred_train.iloc[:, 1]
	#y_pred_all1 = y_pred_all.iloc[:, 0] - y_pred_all.iloc[:, 1]
	y1 = y.iloc[:, 0] - y.iloc[:, 1]

	error = mean_squared_error(y_test1, y_pred1)
	error_train = mean_squared_error(y_train1, y_pred_train1)

	print(f"Mean Squared Error for {model_name}: {error}")
	print(f"Mean Squared Error for {model_name} on training data: {error_train}")
	print(f"R2 Score: {fit_model.score(testset[0], testset[1])}")
	
	dt = pd.DataFrame({'0':X_test1,'1':y_pred1, '2':y_test1.iloc[:, 0]})
	dt2 = pd.DataFrame({'0':X_train1,'1':y_pred_train1, '2':y_train1.iloc[:, 0]})
	dt1 = pd.DataFrame({'x':X1, 'y':y1, 'y_pred':y_pred_all[:, 0]-y_pred_all[:, 1]})
	dt.sort_values(by='0', inplace=True)
	dt1.sort_values(by='x', inplace=True)
	dt2.sort_values(by='0', inplace=True)

	# get more error statistics
	errors = (dt1['y']-dt1['y_pred'])
	dt1['errors'] = errors
	rmse=np.sqrt((dt1['errors']**2).mean())
	print("Root Mean Squared Error for dataset:", rmse)
	nash_sutcliffe=1-(rmse/dt1['y'].std())**2
	print("Nash Sutcliffe Score:", nash_sutcliffe)

	# plot the training data
	plt.scatter(x=X_train1, y=y_train1, color='blue', label='Training Samples')
	plt.scatter(x=X_test1, y=y_test1, color='black', label='Testing Samples')

	# plot the line of best fit
	plt.plot(dt1.loc[:, 'x'], dt1.loc[:, 'y_pred'], color='red', label='Regression Line of Best Fit')
	
	# add axis lines
	plt.axvline(x=-2)
	plt.axvline(x=2)
	plt.axhline(y=-2)
	plt.axhline(y=2)

	# plot the error
	plt.fill_between(dt1['x'], dt1['y_pred']+error, dt1['y_pred']-error, color='red', alpha=.2, label=f'Error: += {error:.2f}°')
	plt.fill_between(dt1['x'], (dt1['y_pred']+.33*error), (dt1['y_pred']-.33*error), color='red', alpha=.4, label='33% Error')
	
	plt.title(f'{model_name} Model Prediction for Planned aHKA from Pre-op aHKA')
	plt.xlabel('Pre-op aHKA')
	plt.ylabel('Planned aHKA')
	plt.legend()
	# plt.savefig('writeup_tex/'+f'{model_name}'.replace(' ','_') +'_regression.png')
	plt.show()
	plt.close()

	
	def show_flat(x_ax, xlbl):
		"""
		Transforms data points to their difference from the regression, flattening the curve
		"""
		plt.scatter(dt1[x_ax], dt1['errors'])
		plt.title('Error Distribution for MLP Regression')
		plt.xlabel(xlbl)
		plt.ylabel('Error (Actual-Predicted) Postop aHKA')
		plt.axhline(y=0, color='red', label='Regression Line of Best Fit')
		plt.legend()
		plt.show()
		plt.close()

	# plot error with respect to the Preoperative aHKA
	#show_flat('x', 'Pre-op aHKA')

	# plot the error with respect to Femoral Rotation: Transverse (External = +, Internal = -) (degrees)
	dt1['FTA'] = df['FTR']
	#show_flat('FTA', 'Pre-op Femoral Transverse Rotation')

	# plot the error with respect to BMI
	dt1['BMI'] = df['BMI']
	#show_flat('BMI', 'Patient BMI')

	# plot the error with respect to Age at Surgery
	dt1['Age'] = df['Age at Surgery']
	#show_flat('Age', 'Patient Age')	


def do_lin():
	# linear regression model
	lin = LinearRegression()
	# fit the model with polynomial features
	poly = PolynomialFeatures(degree=3) # sigmoidal or tanh or something
	X_train_poly = poly.fit_transform(X_train_scalar)
	X_test_poly = poly.transform(X_test_scalar)
	x_poly = poly.transform(X_scalar)
	lin.fit(X_train_poly, y_train)

	test_model(lin, 
			"degree 3 polynomial",
				testset=(X_test_poly, y_test),
				trainset=(X_train_poly, y_train),
				x_data = x_poly)
	
	
# feed-forward neural network model
def do_mlp(train=True):
	if train:
		mlp = MLPRegressor()
		clf = GridSearchCV(mlp, {'hidden_layer_sizes': [(16,8,4)], #(16, 8, 4)
								'max_iter': [2000], #2000
								'solver': ['sgd'], #sgd
								'activation': ['tanh'], #tanh
								'batch_size': [4], #4
								'learning_rate': ['adaptive'], #'adaptive'
								'early_stopping':[False], #False
								'validation_fraction':[0.05], #0.05
								'random_state': [42]
								})
		clf.fit(X_train_normalized, y_train_norm)
		best = clf.best_estimator_
		if 'mlp.h5' in os.listdir(config.pretrained_path):
			st = config.pretrained_path + 'mlp0.h5'
		else:
			st = config.model_path
		pickle.dump(best, open(st, 'wb'))
	else:
		try:
			f = open(config.model_path, 'rb')
			best=pickle.load(f)
		except:
			print(f"Couldn't find pretrained model path {config.model_path}, training anyways")
			do_mlp(train=True)
			exit()
	
	# get the mean squared error, mean absolute error, and root mean squared error over time
	loss = np.array(best.loss_curve_)
	epoch = np.arange(1, loss.shape[0]+1, 1)
	s=''
	for i, (k, v) in enumerate(zip(best.get_params().keys(), best.get_params().values())):
		s+=f'{k:<20} : {str(v):<20}'
		if (i+1)%3==0:
			print(s)
			s=''
	plt.plot(epoch, loss)
	plt.title("MLP Regressor Loss Curve")
	plt.xlabel("Training Epoch")
	plt.ylabel("Mean Squared Error")
	plt.tight_layout()
	plt.show()
	plt.close()

	test_model(best, 
			"neural network",
				testset=(X_test_normalized, y_test_norm),
				trainset=(X_train_normalized, y_train_norm),
				x_data = X_normalized)


# Support Vector Machine model
def do_svm(train=True):
	if train:
		svm = NuSVR()
		clf = GridSearchCV(svm, {'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
								'kernel': ['rbf', 'sigmoid'],
								'degree': [2, 3, 4],
								'gamma': ['scale', 'auto']
								})
		clf.fit(X_train_scalar, y_train)
		best = clf.best_estimator_
		pickle.dump(best, open('support_vector_machine.h5', 'wb'))
	else:
		if 'support_vector_machine.h5' not in os.listdir():
			print("Couldn't find pretrained model of name 'support_vector_machine.h5', training anyways")
			do_svm(train=True)
			exit()
		else:
			best=pickle.load(open('support_vector_machine.h5', 'rb'))

	test_model(best,
			"svm",
				testset=(X_test_scalar, y_test),
				trainset=(X_train_scalar, y_train),
				x_data = X_scalar)

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
				trainset=(X_train_scalar, y_train),
				x_data = X_scalar)

if __name__ == '__main__':
	# do_lin()
	do_mlp(train=False) # training takes a long time to train (R.I.P. YOUR CPU)
	# do_svm(train=True)
	# do_gaus()
