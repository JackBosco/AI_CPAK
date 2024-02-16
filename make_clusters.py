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

print(data)

# Remove the non-preop data from the clustering
preop = data.drop(labels=["Planned aHKA (Varus < -2º, Valgus > 2º)", "Planned JLO (Apex Proximal > 183º, Apex Distal < 177º)"], axis=1)

colors = ['black', 'blue', 'red', 'green', 'grey', 'brown']

# STANDARDIZE with the Standard Scalar
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scalar = StandardScaler()
norm_data = scalar.fit_transform(preop)

# get data from the kmeans
kmeans = KMeans(n_clusters=N_CLUSTERS, max_iter=1000)
kmeans.fit(norm_data)
clusters = kmeans.predict(norm_data)
data['cluster'] = clusters

# make data visualization with the clusters
fig, ax = plt.subplots(nrows=3, figsize = (8,8), sharex='col', sharey='all')

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
	ci = data.loc[data["cluster"]==i, :]
	for x1, y1, x2, y2 in zip(ci.iloc[:, 0], ci.iloc[:, 1], ci.iloc[:, 2], ci.iloc[:, 3]):
		# draw a line from preop to postop on ax[0]
		ax[2].annotate(xy=(x1, y1), xytext=(x2, y2), arrowprops={'arrowstyle':'<-', 'color':color, "alpha":.2}, text=None)
	

ax[0].invert_yaxis()
for a in ax:
	a.axhline(y=177)
	a.axhline(y=183)
	a.axvline(x=-2)
	a.axvline(x=2)
	a.set_xlabel("aHKA (Varus < -2º, Valgus > 2º)")
	a.set_ylabel("JLO")
	# rotate the ylabel by 45 degrees
	a.yaxis.label.set_rotation(45)
	a.legend()

ax[0].set_title("Pre-Op Alignment")
ax[1].set_title("Post-Op Alignment")
ax[2].set_title("Pre-Op to Post-Op Alignment")
fig.suptitle("Clusters of Pre-Op and Post-Op Morphologies with KMeans\n"+
			 "[n_clusters={0} ".format(N_CLUSTERS) + ' '.join(options) + ']')
plt.show()
