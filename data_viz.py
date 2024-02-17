import config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


try:
	f = open(config.treated_path, 'r')
	f.close()
except:
	raise Exception("morphologies.csv not found in ./treated directory. This file is not part of the standard repo. See ./treated/README.md for more details")
df = pd.read_csv(config.treated_path, index_col=0)

cols = list(df.columns)

preop_hka, preop_jlo = df.loc[:, cols[0]], df.loc[:, cols[1]]
postop_hka, postop_jlo = df.loc[:, cols[2]], df.loc[:, cols[3]]

# ==========plot preop data================
fig, ax = plt.subplots(nrows=2, figsize = (8,8), sharex='col', sharey='all')

#plotting black for preop
ax[0].scatter(x=preop_hka, 
		   y=preop_jlo,
			c='black', label='pre-op',) 

#plotting blue for postop
ax[0].scatter(x=postop_hka,
		   y=postop_jlo,
		   c='blue', label='post-op')


preop_verus = df.loc[df["Pre-op Morphology"] % 3 == 1, cols[0]:cols[1]]
avg_preop_verus = np.average(preop_verus[cols[0]]), np.average(preop_verus[cols[1]])

postop_verus = df.loc[df["Pre-op Morphology"] % 3 == 1, cols[2]:cols[3]]
avg_postop_verus = np.average(postop_verus[cols[2]]), np.average(postop_verus[cols[3]])

postop_aligned = df.loc[df["Pre-op Morphology"] % 3 == 2, cols[2]:cols[3]]
avg_postop_aligned = np.average(postop_aligned[cols[2]]), np.average(postop_aligned[cols[3]])

preop_aligned = df.loc[df["Pre-op Morphology"] % 3 == 2, cols[0]:cols[1]]
avg_preop_aligned = np.average(preop_aligned[cols[0]]), np.average(preop_aligned[cols[1]])

preop_valgus = df.loc[df["Pre-op Morphology"] % 3 == 0, cols[0]:cols[1]]
avg_preop_valgus = np.average(preop_valgus[cols[0]]), np.average(preop_valgus[cols[1]])

postop_valgus = df.loc[df["Pre-op Morphology"] % 3 == 0, cols[2]:cols[3]]
avg_postop_valgus = np.average(postop_valgus[cols[2]]), np.average(postop_valgus[cols[3]])


ax[1].scatter(avg_postop_verus[0], avg_postop_verus[1], c='purple', label='Average for varus preop group')
ax[1].annotate('Postop', avg_postop_verus)
ax[1].scatter(avg_preop_verus[0], avg_preop_verus[1], c='purple')
ax[1].annotate('Preop', avg_preop_verus)
ax[1].scatter(avg_preop_aligned[0], avg_preop_aligned[1], c='brown', label='Average for aligned preop group')
ax[1].annotate('Preop', avg_preop_aligned)
ax[1].scatter(avg_postop_aligned[0], avg_postop_aligned[1], c='brown')
ax[1].annotate('Postop', avg_postop_aligned)
ax[1].scatter(avg_preop_valgus[0], avg_preop_valgus[1], c='green', label='Average for valgus preop group')
ax[1].annotate('Preop', avg_preop_valgus)
ax[1].scatter(avg_postop_valgus[0], avg_postop_valgus[1], c='green')
ax[1].annotate('Postop', avg_postop_valgus)

#plot the lines
ax[0].invert_yaxis()
for a in ax:
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
	a.set_ylabel("JLO")
	a.yaxis.label.set_rotation(45)

#axis titles
ax[0].set_title('Pre and Post op Knee Alignment Morphologies')
ax[1].set_title('Average Pre and Post op Knee Alignment Morphologies')

ax[0].legend()
ax[1].legend()
#ax[1].legend(loc='lower right')
plt.show()
