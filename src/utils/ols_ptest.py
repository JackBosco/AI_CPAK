def regp(df, reg_type='linear', y=None, x_num=[], x_cat=[]):
	import pandas as pd
	import statsmodels.api as sm
	data=df.copy()
	xs=x_num
	if y not in data:
		print('')
		print('"'+y+'"'+' is not in dataframe. Please input again.')
		quit()
	for x in x_num:
		if x not in data:
			print('')
			print('"'+x+'"'+' is not in dataframe.')
			quit()
	for x in x_cat:
		if x not in data:
			print('')
			print('"'+x+'"'+' is not in dataframe.')
			quit()
	used_columns=(x_num+x_cat).copy()
	used_columns.append(y)
	data=data[used_columns]
	data=data.dropna()
	for dummy in x_cat:
		unique=list(data[dummy].unique())
		del unique[-1]
		data=pd.get_dummies(data,columns=[dummy],dtype=int)
		for unique_val in unique:
			xs.append(str(dummy)+'_'+str(unique_val))
	if reg_type=='linear':
		return sm.OLS(data[y],sm.add_constant(data[xs])).fit()
	elif reg_type=='logistic':
		return sm.Logit(data[y],sm.add_constant(data[xs])).fit()
