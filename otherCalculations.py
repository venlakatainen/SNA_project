import pandas as pd
import statistics
from scipy.stats import kurtosis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw
from collections import Counter
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy.stats as st
from collections import defaultdict

df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])

# most active followers
f_counts = df.Follower.value_counts().rename_axis('Follower').reset_index(name='Frequency')

# top 10
f_counts[0:10]


# plot in log scale
plt.hist(np.log10(1+f_counts.Frequency),100)
plt.yscale('log')
plt.grid()
plt.title('Distribution of following count')
plt.xlabel('log10(1+counts)')
plt.show()

# most followed nodes
t_counts = df.Target.value_counts().rename_axis('Target').reset_index(name='Frequency')

# top 10
t_counts[0:10]

# plot in log scale
plt.hist(np.log10(1+t_counts.Frequency),100)
plt.yscale('log')
plt.grid()
plt.title('Distribution of target count')
plt.xlabel('log10(1+counts)')
plt.show()

# nodes around the most active follower
follower = 3493
df_select = df[df.Follower==follower]
df_select

# create graph
G_most_active = nx.from_pandas_edgelist(df_select, 'Follower', 'Target', create_using=nx.DiGraph())

# and plot
nx.draw(G_most_active, with_labels=True, node_size=1000, alpha=0.5, arrows=True)
plt.title('Targets that ' + str(follower) + ' follows')
plt.show()