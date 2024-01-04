"""
Jack Bosco
treat_data.py

Treating the MAKO data to remove unuseful data and compress and discretize useful data
"""

import pandas as pd
import sys

df = pd.read_excel("raw/mako_data.xlsx", header=1)
col_headers = list(df.columns)

# TODO change name of the femoral and tibial rotation for the last edited screen

df.rename(columns={"HAS SUMMARY IMAGE (YES=1,NO=0)": "has_summary_image"}, inplace=True)

summary = df[df["has_summary_image"] == 1].copy().set_index(df.columns[0])

planned_headers = {'jlo': 'Femoral Rotation: Coronal (Varus = +, Valgus = -) (degrees)', #not correct
				   'hka' : 'Tibial Rotation: Coronal (Varus = +, Valgus = -) (degrees)'
				}
preop_headers = {'hka': 'Joint Line: aHKA (Varus = +, Valgus = -) (degrees)', # mult by -1
				 'jlo': 'Joint Line: JLO (degrees)' # subtract 180
				}

summary.dropna(axis='index', subset=list(preop_headers.values()) + list(planned_headers.values()), inplace=True) # drops rows without planned and postop hka, jlo

out = summary.loc[:, list(planned_headers.values()) + list(preop_headers.values())]

def get_morphology(jlo, hka, jlo_thresh=3, hka_thresh=2): #
	morph_table = [	[1, 2, 3],
					[4, 5, 6],
					[7, 8, 9]]

	if -1 * hka_thresh < hka and hka < hka_thresh:
		hka_num = 1
	elif hka > hka_thresh:
		hka_num = 0
	else: 
		hka_num = 2
	
	if -1 * jlo_thresh < jlo and jlo < jlo_thresh:
		jlo_num = 1
	elif jlo > jlo_thresh:
		jlo_num = 2
	else: 
		jlo_num = 0

	return morph_table[jlo_num][hka_num]

planned_morphs = out.apply(lambda x: get_morphology(x[planned_headers['jlo']], x[planned_headers['hka']]), axis=1)
preop_morphs = out.apply(lambda x: get_morphology(x[preop_headers['jlo']]-180, x[preop_headers['hka']]*-1), axis=1)

out["Planned Morphology"] = planned_morphs
out["Pre-op Morphology"] = preop_morphs

out.to_csv('treated/morphologies.csv')