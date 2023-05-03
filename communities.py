import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
plt.style.use('fivethirtyeight')

## Network
import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import pylab as plt 
from itertools import count 
from operator import itemgetter 
from networkx.algorithms.community import modularity 
from networkx.algorithms.connectivity import k_components
from networkx.algorithms.community import label_propagation_communities
import pylab

df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])

#put the node id of the node with the highest node betweenness here
node = 4018

#Subset
#df = df[node - 2000 : node + 2000]
df = df[0 : 10000]

#Creating the graph
pd.set_option('display.precision',10)
G = nx.from_pandas_edgelist(df, 'Follower', 'Target', create_using = nx.Graph())

#Label propagation
communities = label_propagation_communities(G)
print(f"Amount of communities: {len(communities)}")
print("Communities:")
print(communities)

#Overlapping
modularity = modularity(G, communities)
print("Modularity:")
print(modularity)

#Cohesion
print("K-components:")
k_components_list = k_components(G)
print(k_components_list)


#Visualization
nodes = G.nodes()
degree = G.degree()
colors = [degree[n] for n in nodes]

pos = nx.kamada_kawai_layout(G)
#cmap = plt.cm.viridis_r
#cmap = plt.cm.Greys

vmin = min(colors)
vmax = max(colors)

fig = plt.figure(figsize = (15,9), dpi=100)

#nx.draw(G,pos,alpha = 0.8, nodelist = nodes, node_color = 'w', node_size = 10, with_labels= False,font_size = 6, width = 0.2, cmap = cmap, edge_color ='yellow')
nx.draw(G,pos,alpha = 0.8, nodelist = nodes, node_color = 'w', node_size = 10, with_labels= False,font_size = 6, width = 0.2, edge_color ='yellow')
fig.set_facecolor('#0B243B')

#plt.legend()
plt.show()