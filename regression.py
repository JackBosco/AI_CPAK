"""
	Regression analysis for the alignment data using linear regression
	Author: Jack Bosco
"""
import os
from sklearn.svm import NuSVR
import config
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import scienceplots

# setting plot display parameters
plt.style.use('science')
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['figure.dpi'] = 300

# read in the data, split into X and y
df = pd.read_csv(config.treated_path, index_col=0)
X = df.loc[:, ['Pre-op mpta', 'Pre-op ldfa']] #1, 6, 7, 8]] # preop hka, preop jlo, age, bmi, femoral transverse rotation
y = df.loc[:, ['Planned MPTA', 'Planned LDFA']] # planned hka

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 75% training, 25% testing, 42 is the answer to everything


# initializer normalizer with min-max scaling
normalizer = MinMaxScaler(feature_range=(-1, 1)) # for inputs
out_normalizer = MinMaxScaler(feature_range=(-1, 1)) # for targets

# fit normalizer to training input data
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test) # normalized testing input data
X_normalized = normalizer.transform(X) # normalizing all input data

# normalize the output data with min-max scaling
y_train_norm = out_normalizer.fit_transform(y_train) # training
y_test_norm = out_normalizer.transform(y_test) # testing
y_norm = out_normalizer.transform(y) # both

# save the normalizers
pickle.dump(normalizer, open(config.norm_path, 'wb'))
pickle.dump(out_normalizer, open(config.de_norm_path, 'wb'))


#print descriptive statistics of the normalized data
def print_norms():
	s = [
		(X_train_normalized, "Normalized Pre-op Training Set")
		, (X_test_normalized, "Normalized Pre-op Testing Set")
		, (X_normalized, "Normalized Entire Pre-op Set")
		, (y_train_norm, "Normalized planned Training Set")
		, (y_test_norm, "Normalized planned Testing Set")
		, (y_norm, "Normalized Entire planned Set")
		]
	def helper(data, msg):
		x_df = pd.DataFrame(data, columns=["MPTA","LDFA"])
		print('\n'+msg)
		print(x_df.describe())
	for d, m in s:
		helper(d,m)
print_norms()

def test_model(fit_model, model_name, testset, trainset, x_data, norm1=out_normalizer):
	"""
	@params
	fit_model: the pre-fitted sklearn model
	model_name: the name for printing purposes. Only matters for the title ofthe scatterplot.
	testset: tuple of (input: label) of the normalized testing data. Passed as a numpy array, but is in same order by redcap id
	trainset: tuple of (input: label) of the normalized training data. Passed as a numpy array, but is in same order by redcap id
	x_data: the entire normalized entire dataset. Passed as a numpy array, but is in same order by redcap id
	"""
	
	# this is some serious spaghettic code, let me explain:
	
	#we predict results from the normalized data, then denormalize the output. This is the testing set.
	y_pred = fit_model.predict(testset[0])
	y_pred = pd.DataFrame(norm1.inverse_transform(y_pred))
	
	#we predict results from the normalized data, then denormalize the output. This is the training set.
	y_pred_train = fit_model.predict(trainset[0])
	y_pred_train = pd.DataFrame(norm1.inverse_transform(y_pred_train))
	
	#we predict results from the normalized data, then denormalize the output. This is all the data.
	y_pred_all = fit_model.predict(x_data)
	y_pred_all = norm1.inverse_transform(y_pred_all)
	
	# this is where we get the aHKA for the testing and training sets
	# we also tranform the data from a numpy array to a pandas dataframe
	y_test1, y_train1 = pd.DataFrame(y_test.iloc[:, 0] - y_test.iloc[:, 1]), pd.DataFrame(y_train.iloc[:, 0] - y_train.iloc[:, 1])

	# turn the data from MPTA and LDFA to aHKA
	# testing [MPTA, LDFA] -> aHKA
	X_test1 = X_test.iloc[:, 0] - X_test.iloc[:, 1]
	
	#training [MPTA, LDFA] -> aHKA
	X_train1 = X_train.iloc[:, 0] - X_train.iloc[:, 1]

	# turn everything that isn't a Pandas Datafram into a Pandas Datafram
	X1 = X.iloc[:, 0] - X.iloc[:, 1]
	y_pred1 = y_pred.iloc[:, 0] - y_pred.iloc[:, 1]
	y_pred_train1 = y_pred_train.iloc[:, 0] - y_pred_train.iloc[:, 1]
	#y_pred_all1 = y_pred_all.iloc[:, 0] - y_pred_all.iloc[:, 1]
	y1 = y.iloc[:, 0] - y.iloc[:, 1]

	# get the mean squared error and print
	error = mean_squared_error(y_test1, y_pred1)
	error_train = mean_squared_error(y_train1, y_pred_train1)
	print(f"Mean Squared Error for {model_name}: {error}")
	print(f"Mean Squared Error for {model_name} on training data: {error_train}")
	print(f"R2 Score: {fit_model.score(testset[0], testset[1])}")
	
	# testing dataset dataframe
	dt = pd.DataFrame({'0':X_test1,'1':y_pred1, '2':y_test1.iloc[:, 0]})

	# training dataset dataframe
	dt2 = pd.DataFrame({'0':X_train1,'1':y_pred_train1, '2':y_train1.iloc[:, 0]})

	# all the data dataframe
	dt1 = pd.DataFrame({'x':X1, 'y':y1, 'y_pred':y_pred_all[:, 0]-y_pred_all[:, 1]})

	# sorting it so it makes the graphs look readable
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
	#plt.fill_between(dt1['x'], (dt1['y_pred']+.33*error), (dt1['y_pred']-.33*error), color='red', alpha=.4, label='33% Error')
	
	# plot the rest
	plt.title(f'NYU Method Prediction of Planned aHKA from Preoperative aHKA')
	plt.xlabel('Preoperative aHKA (degrees)')
	plt.ylabel('Planned aHKA (degrees)')
	plt.legend()
	# plt.savefig('Figure 3')#'writeup_tex/'+f'{model_name}'.replace(' ','_') +'_regression.png') #this can cause problems
	plt.show()
	plt.close()

	
	# all of this is turned off, we are basically done
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

	
# feed-forward neural network model
def do_mlp(train=False):
	"""
	@params 
	train: whether or not you want to train a fresh model. If there is no pretrained model available, it will train anyways
	"""
	if train:
		mlp = MLPRegressor()
		# this can try a bunch of stuff, but only does once
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

		# fit the model
		clf.fit(X_train_normalized, y_train_norm)
		# get the best model (per the training data, can be overfit)
		best = clf.best_estimator_
		# save it, if not there
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
	
	# plot the loss curve
	plt.plot(epoch, loss)
	plt.title("MLP Regressor Loss Curve")
	plt.xlabel("Training Epoch")
	plt.ylabel("Mean Squared Error")
	plt.tight_layout()
	# plt.savefig("Figure 2")
	plt.show()
	plt.close()

	test_model(best, 
			"neural network",
				testset=(X_test_normalized, y_test_norm),
				trainset=(X_train_normalized, y_train_norm),
				x_data = X_normalized)


if __name__ == '__main__':
	do_mlp(train=False) # training takes a long time to train (R.I.P. YOUR CPU)
