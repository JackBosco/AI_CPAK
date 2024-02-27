"""
Jack Bosco
"""

import config
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

try:
	f = open(config.treated_path, 'r')
	f.close()
except:
	raise Exception("morphologies.csv not found in ./treated directory. This file is not part of the standard repo. See ./treated/README.md for more details")
df = pd.read_csv(config.treated_path, index_col=0)
data = df.iloc[:, :4]

# optional arguments for including more elements in the clustering
N_CLUSTERS = 3
options = [f'n_clusters={N_CLUSTERS}']
if 'age' in sys.argv:
	options.append('age')
	data['age'] = df['Age at Surgery']
if 'bmi' in sys.argv:
	options.append('bmi')
	data['bmi'] = df['BMI']
if 'FTR' in sys.argv:
	options.append('femoral transverse rotation')
	data['FTR'] = df.loc[:, 'FTR']
if 'sex' in sys.argv:
	data['ismale'] = df.loc[:, 'sex_0']
	data['isfemale'] = df.loc[:, 'sex_1']
	options.append('sex')
if 'nclusters' in sys.argv:
	N_CLUSTERS = int(sys.argv[sys.argv.index('nclusters')+1])
	options[0] = f'n_clusters={N_CLUSTERS}'


colors = ['black', 'blue', 'red', 'green', 'grey', 'brown']

# STANDARDIZE with the Standard Scalar
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scalar = StandardScaler()
norm_data = scalar.fit_transform(data)

# get data from the kmeans
kmeans = KMeans(n_clusters=N_CLUSTERS, max_iter=1000)
kmeans.fit(norm_data)
clusters = kmeans.predict(norm_data)
data['cluster'] = clusters


# make data visualization with the clusters
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

ax = [[ax1, ax2], [ax3, ax4]]

#plotting black for preop
for i, color in zip(range(N_CLUSTERS), colors):
	cluster = data.loc[data['cluster'] == i]
	ax[0][0].scatter(x=cluster.iloc[:, 0],
				y=cluster.iloc[:, 1],
				c=color, label="cluster " + str(i),
				alpha=0.8)
	ax[0][1].scatter(x=cluster.iloc[:, 2],
				y=cluster.iloc[:, 3],
				c=color, label="cluster " + str(i),
				alpha=0.8)
	ci = data.loc[data["cluster"]==i, :]
	for x1, y1, x2, y2 in zip(ci.iloc[:, 0], ci.iloc[:, 1], ci.iloc[:, 2], ci.iloc[:, 3]):
		# draw a line from preop to postop on ax[0]
		ax[1][0].annotate(xy=(x1, y1), xytext=(x2, y2), arrowprops={'arrowstyle':'<-', 'color':color, "alpha":.2}, text=None)
	ax[1][1].annotate(xy=(np.average(ci.iloc[:, 0]), np.average(ci.iloc[:, 1]))
			, xytext=(np.average(cluster.iloc[:, 2]), np.average(cluster.iloc[:, 3]))
			, arrowprops={'color':color, 'alpha':1, 'arrowstyle':'<|-'}, text=None)
	ax[1][1].arrow(0, 180, 0, 0, color=color, label=f'Average change for cluster {i}')
	ax[1][0].arrow(0, 180, 0, 0, color=color, label=f'Pre-op→Post-op Change for cluster {i}')
	
x, y = ax1.get_xlim(), ax1.get_ylim()
for a in (ax[0][0], ax[0][1], ax[1][0], ax[1][1]):
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
	a.set_ylabel("JLO")
	# rotate the ylabel by 45 degrees
	a.yaxis.label.set_rotation(45)
	a.set_xlim(x[0], x[1])
	a.set_ylim(y[0], y[1])
	a.invert_yaxis()
	a.legend()

ax[0][0].set_title("Pre-Op Alignment")
ax[0][1].set_title("Post-Op Alignment")
ax[1][0].set_title("Pre-Op to Post-Op Alignment")
ax[1][1].set_title("Average Pre-Op to Post-Op Alignment")
print("Clusters of Pre-Op and Post-Op Morphologies with KMeans\n"+
			 "["+', '.join(options) + ']')
if 'save' in sys.argv:
	plt.savefig('writeup_tex/clusters.png')
plt.show()
