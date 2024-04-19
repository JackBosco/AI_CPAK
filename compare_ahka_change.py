import pandas as pd
from ols_ptest import regp

df=pd.read_csv("data/treated/morphologies.csv", index_col=0)

df['aHKA morph'] = df['Pre-op Morphology'] % 3
df['ahka change'] = (df['Planned aHKA (Varus < -2º, Valgus > 2º)'] - df['Pre-op aHKA (Varus < -2º, Valgus > 2º)']).abs()
no_aligned = df.drop(df.loc[df['aHKA morph']==1, :].index)

results = regp(no_aligned, reg_type='linear', y='ahka change', x_cat=['aHKA morph'])
print(f"Extended p-value: {results.f_pvalue}")

import sys
if len(sys.argv)==1:
	print(results.summary())
else:
	print(results.summary().as_latex())
