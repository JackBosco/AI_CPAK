"""
Jack Bosco
"""

from click import group
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

try:
	f = open('treated/morphologies.csv', 'r')
	f.close()
except:
	raise Exception("morphologies.csv not found in ./treated directory. This file is not part of the standard repo. See ./treated/README.md for more details")
df = pd.read_csv('treated/morphologies.csv')
data = df.iloc[:, 1:5]
N_CLUSTERS = 3
colors = ['black', 'blue', 'red', 'green']

# get data from the kmeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
clusters = kmeans.predict(data)
data['cluster'] = clusters

# make data visualization with the clusters
fig, ax = plt.subplots(nrows=2, figsize = (8,8), sharex='col', sharey='all')

#plotting black for preop
for i, color in zip(range(N_CLUSTERS), colors):
	cluster = data.loc[data['cluster'] == i]
	ax[0].scatter(x=cluster.iloc[:, 0],
				y=cluster.iloc[:, 1],
				c=color, label="cluster " + str(i))
	ax[1].scatter(x=cluster.iloc[:, 2],
				y=cluster.iloc[:, 3],
				c=color, label="cluster " + str(i))

ax[0].invert_yaxis()
for a in ax:
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
	a.set_ylabel("JLO (Apex Distal > 183º, Apex Proximal < 177º)")
	a.legend()

ax[0].set_title("Pre-Op Alignment")
ax[1].set_title("Post-Op Alignment")
fig.suptitle("Clusters of Pre-Op and Post-Op Morphologies with KMeans n_clusters=3")
plt.show()