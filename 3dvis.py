from tkinter import Y
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge, ARDRegression

df = pd.read_csv('treated/morphologies.csv', index_col=0)
X = df.loc[:, 'Pre-op aHKA (Varus < -2º, Valgus > 2º)']
Z = df.loc[:, 'Pre-op JLO (Apex Proximal > 183º, Apex Distal < 177º)']
Y = df.loc[:, 'Planned aHKA (Varus < -2º, Valgus > 2º)']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Z, Y, c='r', marker='o')
ax.set_zlabel('Planned aHKA')
plt.xlabel('Pre-op aHKA')
plt.ylabel('Pre-op JLO')
plt.show()