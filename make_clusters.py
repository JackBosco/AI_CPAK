"""
Jack Bosco
"""

from click import group
import pandas as pd
from pyparsing import col
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import StandardScaler

try:
	f = open('treated/morphologies.csv', 'r')
	f.close()
except:
	raise Exception("morphologies.csv not found in ./treated directory. This file is not part of the standard repo. See ./treated/README.md for more details")
df = pd.read_csv('treated/morphologies.csv', index_col=0)
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

# remove outliers in JLO and aHKA for preop using IQR
# q1 = data['Pre-op aHKA (Varus < -2º, Valgus > 2º)'].quantile(0.25)
# q3 = data['Pre-op aHKA (Varus < -2º, Valgus > 2º)'].quantile(0.75)
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
# data = data[(data['Pre-op aHKA (Varus < -2º, Valgus > 2º)'] >= lower_bound) & (data['Pre-op aHKA (Varus < -2º, Valgus > 2º)'] <= upper_bound)]

# q1 = data['Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)'].quantile(0.25)
# q3 = data['Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)'].quantile(0.75)
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr
# data = data[(data['Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)'] >= lower_bound) & (data['Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)'] <= upper_bound)]

# STANDARDIZE with the Standard Scalar
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scalar = StandardScaler()
norm_data = scalar.fit_transform(data)

# drop the JLOs from the normalized data
norm_data.drop(columns=['Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)', 'Planned JLO (Apex Proximal > 183º, Apex Distal < 177º)'], inplace=True)

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
