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

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


#plotting black for preop
ax1.scatter(x=preop_hka, 
		   y=preop_jlo,
			c='black', label='pre-op',) 

#plotting blue for postop
ax1.scatter(x=postop_hka,
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

# varus
ax2.scatter(avg_postop_verus[0], avg_postop_verus[1], c='purple', label='Average for varus preop→postop group')
ax2.scatter(avg_preop_verus[0], avg_preop_verus[1], c='purple')
ax2.annotate(xy=avg_preop_verus, xytext=avg_postop_verus, arrowprops={'arrowstyle':'<|-', 'color':'purple', "alpha":1}, text=None)

#aligned
ax2.scatter(avg_preop_aligned[0], avg_preop_aligned[1], c='brown', label='Average for aligned preop→postop group')
ax2.scatter(avg_postop_aligned[0], avg_postop_aligned[1], c='brown')
ax2.annotate(xy=avg_preop_aligned, xytext=avg_postop_aligned, arrowprops={'arrowstyle':'<|-', 'color':'brown', "alpha":1}, text=None)

#valgus
ax2.scatter(avg_preop_valgus[0], avg_preop_valgus[1], c='green', label='Average for valgus preop→postop group')
ax2.scatter(avg_postop_valgus[0], avg_postop_valgus[1], c='green')
ax2.annotate(xy=avg_preop_valgus, xytext=avg_postop_valgus, arrowprops={'arrowstyle':'<|-', 'color':'green', "alpha":1}, text=None)

x, y = ax1.get_xlim(), ax1.get_ylim()
ax2.set_ylim(ymin=y[0], ymax=y[1])
ax2.set_xlim(xmin=x[0], xmax=x[1])
fig1.set_size_inches(5,5)
fig2.set_size_inches(5,5)

#plot the lines
for a in [ax1, ax2]:
	a.invert_yaxis()
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA")
	a.set_ylabel("JLO")
	a.yaxis.label.set_rotation(45)

#axis titles
ax1.set_title('Pre and Post op Knee Alignment Morphologies')
ax2.set_title('Average Pre and Post op Knee Alignment Morphologies')

ax1.legend()
ax2.legend()
#ax2.legend(loc='lower right')

#fig2.add_axes(ax2)
plt.show()
