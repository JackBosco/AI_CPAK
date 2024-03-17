"""
Main user interface for running the model
Author: Jack Bosco
"""

from sklearn.neural_network import MLPRegressor
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config

normalizer=pickle.load(open(config.norm_path, 'rb'))
model=pickle.load(open(config.model_path, 'rb'))
st="{:=^75}"
#data = np.array([[0.0, 0.0]],dtype=float)
data=pd.DataFrame(data={'Pre-op mpta':[], 'Pre-op ldfa':[]}, dtype=float)

while True:
	print('\n'+st.format('[ UI ]'), end='\n\n')
	mpta = float(input('Enter MPTA:\n90 - [Tibial Coronal Rotation (Varus = +, Valgus = -) (degrees)] = '))
	ldfa = float(input('Enter LDFA:\n90 + [Femoral Coronal Rotation (Varus = +, Valgus = -) (degrees)] = '))

	print('\n'+st.format('[ Results ]'),end='\n')
	print("Pre-op:")
	print(f"{'aHKA (Varus < -2º, Valgus > 2º)':<50s}: {mpta-ldfa:> 10.4f}")
	print(f"{'JLO (Apex Proximal > 183º, Apex Distal < 177º)':<50s}: {mpta+ldfa:> 10.4f}")

	data['Pre-op mpta'] = [mpta]
	data['Pre-op ldfa'] = [ldfa]
	inpts = normalizer.transform(data)
	
	preds=model.predict(inpts)
	o_mpta, o_ldfa = preds[-1]
	print("\nPredicted:")
	print(f"{'MPTA':<50s}: {o_mpta:>10.4f}")
	print(f"{'LDFA':<50s}: {o_ldfa:>10.4f}")
	print(f"{'aHKA':<50s}: {o_mpta-o_ldfa:> 10.4f}")
	print(f"{'JLO':<50s}: {o_mpta+o_ldfa:> 10.4f}")