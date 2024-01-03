"""
Jack Bosco
treat_data.py

Treating the MAKO data to remove unuseful data and compress and discretize useful data
"""

import pandas as pd
import sys

df = pd.read_excel("raw/mako_data.xlsx", header=1)

df.rename(columns={"HAS SUMMARY IMAGE (YES=1,NO=0)": "has_summary_image"}, inplace=True)
df.rename(columns={"HAS LAST EDITED PLAN IMAGE (YES=1,NO=0)": "has_last_image"}, inplace=True)

# preop       : dataframe where each entry has the preop image
# preop_postop : dataframe where each entry has both preop and postop image
summary = (df["has_summary_image"] == 1).copy().set_index(df.columns[0])
last_only = ((df["has_last_image"] == 1 ) & (df["has_summary_image"] == 0)).copy().set_index(df.columns[0])

# drop the null columns from the preop only dataset
col_names = list(df.columns)
idx = col_names.index("has_postop_image")
preop.drop(columns=col_names[idx:]+["has_postop_image"], axis=1, inplace=True)

# drop the has_image label
preop_postop.drop(columns=["has_preop_image", "has_postop_image"], axis=1, inplace=True)

print("\nPreop only Dataset: \n", preop)
print("\nPreop and Postop Dataset \n", preop_postop)

if len(sys.argv) > 1 and sys.argv[1] == "w":
	preop.to_csv("treated/preop_only_data.csv")
	preop_postop.to_csv("treated/preop_postop_data.csv")
