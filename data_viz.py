from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("treated/morphologies.csv", index_col=0)

cols = list(df.columns)

preop_hka, preop_jlo = df.loc[:, cols[0]], df.loc[:, cols[1]]
postop_hka, postop_jlo = df.loc[:, cols[2]], df.loc[:, cols[3]]

# ==========plot preop data================
fig, ax = plt.subplots(figsize = (8,8))

#plotting black for preop
ax.scatter(x=preop_hka, 
		   y=preop_jlo,
			c='black', label='pre-op',) 

#plotting blue for postop
ax.scatter(x=postop_hka,
		   y=postop_jlo,
		   c='blue', label='post-op')

ax.invert_yaxis()

max_x = max(max(preop_hka), max(postop_hka))
min_x = min(min(preop_hka), min(postop_hka))

#plot the lines
plt.axhline(y=177)
plt.axhline(y=183)
plt.axvline(x=-2)
plt.axvline(x=2)

#axis titles
ax.set_title('Pre and Post op Knee Alignment Morphologies')
ax.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
ax.set_ylabel("JLO (Apex Distal > 183º, Apex Proximal < 177º)")

plt.legend()
plt.show()