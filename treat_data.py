"""
Jack Bosco
treat_data.py

Treating the MAKO data to remove unuseful data and compress and discretize useful data
"""

import pandas as pd
"""
Jack Bosco
Treat the data
"""

import config
try:
	temp = open(config.raw_path, 'r')
	temp.close()
except:
	raise Exception("mako_data.xlsx not present in ./raw directory. This file is not part of the standard repo. See ./raw/README.md for more details")
df = pd.read_excel(config.raw_path, header=1, index_col=0)

df.rename(columns={"HAS SUMMARY IMAGE (YES=1,NO=0)": "has_summary_image"}, inplace=True)

summary = df[df["has_summary_image"] == 1].copy()

# declare headers 

planned_headers = {	'ldfa': 'Planned LDFA',
				   	'mpta' : 'Planned MPTA',
					'hka' : 'Planned aHKA (Varus < -2º, Valgus > 2º)',
					'jlo' : 'Planned JLO (Apex Proximal > 183º, Apex Distal < 177º)'
				}
preop_headers = {'hka': 'Pre-op aHKA (Varus < -2º, Valgus > 2º)', # mult by -1
				 'jlo': 'Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)', # subtract 180
				 'mpta': 'Pre-op mpta',
				 'ldfa': 'Pre-op ldfa'
				}
pre_mpta_old = 'Joint Line: MPTA (degrees)'
pre_ldfa_old = 'Joint Line: LDFA (degrees)'
old_hka = 'Joint Line: aHKA (Varus = +, Valgus = -) (degrees)' 
old_jlo = 'Joint Line: JLO (degrees)'

post_ldfa_old = 'Femoral Rotation: Coronal (Varus = +, Valgus = -) (degrees)'
post_mpta_old = 'Tibial Rotation: Coronal (Varus = +, Valgus = -) (degrees).1'

# rename the bad preop hka and jlo headers to include pre-op
summary.rename(columns={old_hka: preop_headers['hka'], old_jlo: preop_headers['jlo']}, inplace=True)

# rename bad postop mpta and jlo headers
summary.rename(columns={
				post_mpta_old: planned_headers['mpta']
				, post_ldfa_old: planned_headers['ldfa']
				}, inplace=True)

# rename the bad preop mpta and jlo headers
summary.rename(columns={
				pre_mpta_old: preop_headers['mpta']
				, pre_ldfa_old: preop_headers['ldfa']
				}, inplace=True)

summary.dropna(axis='index', 
			   subset=list(preop_headers.values()) +
					[planned_headers['ldfa'], planned_headers['mpta']] ,
					inplace=True) # drops rows without planned and postop hka, jlo, ldfa, mpta

#==================== preop ======================
# hka, jlo, mpta, ldfa for pre-op
out = summary.loc[:, list(preop_headers.values()) + ['Age at Surgery', 'BMI']]
# hka *= -1
out[preop_headers['hka']] *= -1

#==================== postop =====================

#out = summary.loc[:, list(planned_headers.values()) + list(preop_headers.values())]

# project ldfa and mpta from summary images
planned_vals = summary.loc[:, [planned_headers['ldfa'], planned_headers['mpta']]]

# MPTA = 90 - summary tibial cronal rotation
planned_vals[planned_headers['mpta']] *= -1
planned_vals[planned_headers['mpta']] += 90

# LDFA = summary femoral coronal rotation + 90
planned_vals[planned_headers['ldfa']] += 90

# add planned ldfa and mpta to output
out[planned_headers['ldfa']] = summary[planned_headers['ldfa']]
out[planned_headers['mpta']] = summary[planned_headers['mpta']]

#hka, jlo for post-op
#aHKA = MPTA - LDFA
out[planned_headers['hka']] = planned_vals[planned_headers['mpta']] - planned_vals[planned_headers['ldfa']]
#JLO = MPTA + LDFA
out[planned_headers['jlo']] = planned_vals[planned_headers['mpta']] + planned_vals[planned_headers['ldfa']]

# =================== move ldfa and mpta to the back =============
for label in [planned_headers['ldfa'], planned_headers['mpta'], preop_headers['ldfa'], preop_headers['mpta']]:
	out[label] = out.pop(label) 


# ===================getting morphologies =======================

def get_morphology(jlo, hka, jlo_thresh=3, hka_thresh=2): #
	morph_table = [	[1, 2, 3],
					[4, 5, 6],
					[7, 8, 9]]

	if hka < -1 * hka_thresh:
		hka_num = 0
	elif hka > hka_thresh:
		hka_num = 2
	else: 
		hka_num = 1
	
	if jlo < -1 * jlo_thresh:
		jlo_num = 0
	elif jlo > jlo_thresh:
		jlo_num = 2
	else: 
		jlo_num = 1

	return morph_table[jlo_num][hka_num]

planned_morphs = out.apply(lambda x: get_morphology(x[planned_headers['jlo']]-180, x[planned_headers['hka']]), axis=1)
preop_morphs = out.apply(lambda x: get_morphology(x[preop_headers['jlo']]-180, x[preop_headers['hka']]), axis=1)

out["Planned Morphology"] = planned_morphs
out["Pre-op Morphology"] = preop_morphs

age, bmi = out.loc[:, 'Age at Surgery'], out.loc[:, 'BMI']
out.drop(columns=['Age at Surgery', 'BMI'], inplace=True)
out['Age at Surgery']= age
out['BMI']= bmi

#==================== add femoral transverse rotation =====================

out = out.join(df.loc[:, 'Femoral Rotation: Transverse (External = +, Internal = -) (degrees)'].dropna(), how='inner')
out.rename(columns={'Femoral Rotation: Transverse (External = +, Internal = -) (degrees)': 'FTR'}, inplace=True)

#================ add sex ======================
sex = pd.get_dummies(df.loc[:, 'Sex (1=female,0=male)'], prefix='sex', dtype=float)
out = out.join(sex, how='inner')

#================ add preop native flection ======================
# print(out.shape)
# flexion_data = pd.read_csv(config.raw_flx_path, index_col=0)
# out = out.join(flexion_data.loc[:, 'Preop Native Flexion (Flexion=+, Extension=-)'], how='inner')
# out.rename(axis=1, mapper={'Preop Native Flexion (Flexion=+, Extension=-)':'Preop Flexion'})
# print(out.shape)
"""
DONT DO THIS: 
the flexion data unioned with the other data is so sparse that we lose a lot% of the dataset
"""

#================ save to csv ======================

out.to_csv(config.treated_path)
print(f"Shape: {out.shape}")