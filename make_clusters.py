"""
Jack Bosco
"""

import config
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler

try:
	f = open(config.treated_path, 'r')
	f.close()
except:
	raise Exception("morphologies.csv not found in ./treated directory. This file is not part of the standard repo. See ./treated/README.md for more details")
df = pd.read_csv(config.treated_path, index_col=0)
data = df.iloc[:, 0:4]

# optional arguments for including more elements in the clustering
options = []
if 'age' in sys.argv:
	options.append('age')
	data['age'] = df['Age at Surgery']
if 'bmi' in sys.argv:
	options.append('bmi')
	data['bmi'] = df['BMI']
if 'clusters' in sys.argv:
	N_CLUSTERS = int(sys.argv[sys.argv.index('clusters')+1])
else:
	N_CLUSTERS = 4
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
fig, ax = plt.subplots(nrows=2, figsize = (8,8), sharex='col', sharey='all')
	

#plotting black for preop
for i, color in zip(range(N_CLUSTERS), colors):
	cluster = data.loc[data['cluster'] == i]
	ax[0].scatter(x=cluster.iloc[:, 0],
				y=cluster.iloc[:, 1],
				c=color, label="cluster " + str(i),
				alpha=0.8)
	ax[1].scatter(x=cluster.iloc[:, 2],
				y=cluster.iloc[:, 3],
				c=color, label="cluster " + str(i),
				alpha=0.8)

ax[0].invert_yaxis()
for a in ax:
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
	a.set_ylabel("JLO (Apex Proximal > 183º, Apex Distal < 177º)")
	a.legend()

ax[0].set_title("Pre-Op Alignment")
ax[1].set_title("Post-Op Alignment")
fig.suptitle("Clusters of Pre-Op and Post-Op Morphologies with KMeans\n"+
			 "[n_clusters={0} ".format(N_CLUSTERS) + ' '.join(options) + ']')
plt.show()
