"""
Jack Bosco
treat_data.py

Treating the MAKO data to remove unuseful data and compress and discretize useful data
"""

import pandas as pd
import sys

try:
	temp = open("raw/mako_data.xlsx", 'r')
	temp.close()
except:
	raise Exception("mako_data.xlsx not present in ./raw directory. This file is not part of the standard repo. See ./raw/README.md for more details")
df = pd.read_excel("raw/mako_data.xlsx", header=1, index_col=0)

df.rename(columns={"HAS SUMMARY IMAGE (YES=1,NO=0)": "has_summary_image"}, inplace=True)

summary = df[df["has_summary_image"] == 1].copy()

# declare headers 

planned_headers = {	'ldfa': 'Femoral Rotation: Coronal (Varus = +, Valgus = -) (degrees)',
				   	'mpta' : 'Tibial Rotation: Coronal (Varus = +, Valgus = -) (degrees).1',
					'hka' : 'Planned aHKA (Varus < -2º, Valgus > 2º)',
					'jlo' : 'Planned JLO (Apex Proximal > 183º, Apex Distal < 177º)'
				}
preop_headers = {'hka': 'Pre-op aHKA (Varus < -2º, Valgus > 2º)', # mult by -1
				 'jlo': 'Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)' # subtract 180
				}
old_hka = 'Joint Line: aHKA (Varus = +, Valgus = -) (degrees)' 
old_jlo = 'Joint Line: JLO (degrees)'

summary.rename(columns={old_hka: preop_headers['hka'], old_jlo: preop_headers['jlo']}, inplace=True)

summary.dropna(axis='index', 
			   subset=list(preop_headers.values()) +
					[planned_headers['ldfa'], planned_headers['mpta']] ,
					inplace=True) # drops rows without planned and postop hka, jlo, ldfa, mpta

#==================== preop ======================
# hka, jlo for pre-op
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

#hka, jlo for post-op
#aHKA = MPTA - LDFA
out[planned_headers['hka']] = planned_vals[planned_headers['mpta']] - planned_vals[planned_headers['ldfa']]
#JLO = MPTA + LDFA
out[planned_headers['jlo']] = planned_vals[planned_headers['mpta']] + planned_vals[planned_headers['ldfa']]

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

out.to_csv('treated/morphologies.csv')
