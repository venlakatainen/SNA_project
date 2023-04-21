import pandas as pd
import statistics
from scipy.stats import kurtosis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw

# read data

def get_twitter_data():
    # read edges
    edges = pd.read_csv("edges.csv")
    # read nodes
    nodes = pd.read_csv("nodes.csv")
    # return nodes and edges
    return edges, nodes


def draw_graph(edgelist):
    G = nx.from_pandas_edgelist(edgelist, 'Follower', 'Target', create_using=nx.DiGraph())
    return G

def get_degrees(graph):
    in_deg = list(graph.in_degree())
    out_deg = list(graph.out_degree())
    avg_deg = []
    
    for i in range(len(in_deg) - 1):
        avg_deg.append((in_deg[i][0], (in_deg[i][1] + out_deg[i][1]) / 2))
    
    in_deg.sort(key = lambda x:x[1])
    out_deg.sort(key = lambda x:x[1])
    avg_deg.sort(key = lambda x:x[1])
    return in_deg, out_deg, avg_deg

def plot_degrees(in_deg, out_deg, avg_deg):
    # in degree
    fig1 = plt.figure("In degree", figsize=(8, 8))
    ax1 = fig1.add_subplot()
    degree_sequence = sorted((d for n, d in in_deg), reverse=True)
    ax1.bar(*np.unique(degree_sequence, return_counts=True), align='center', width=0.4)
    ax1.set_title("In Degree")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    # out degree
    fig2 = plt.figure("Out degree", figsize=(8, 8))
    ax2 = fig2.add_subplot()
    degree_sequence = sorted((d for n, d in out_deg), reverse=True)
    ax2.bar(*np.unique(degree_sequence, return_counts=True), align='center', width=0.4)
    ax2.set_title("Out degree")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    # average degree
    fig3 = plt.figure("Average degree", figsize=(8, 8))
    ax3 = fig3.add_subplot()
    degree_sequence = sorted((d for n, d in avg_deg), reverse=True)
    ax3.bar(*np.unique(degree_sequence, return_counts=True), align='center', width=0.4)
    ax3.set_title("Average Degree")
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("# of Nodes")

def draw_power_laws(in_deg, out_deg, avg_deg):
    pass


# read csv
df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])

# select subset
df_sub = df.sample(1000, random_state=987)

# build graph from data frame
G = draw_graph(df_sub)

# create graph from edges and plot
nx.draw(G, with_labels=True, node_size=1000, alpha=0.5, arrows=True)
plt.title('1000 node subgraph')

in_deg, out_deg, avg_deg = get_degrees(G)
plot_degrees(in_deg, out_deg, avg_deg)

#degree_sequence = sorted((d for n, d in out_deg), reverse=True)
#fit = powerlaw.Fit(degree_sequence)
"""
Calculating best minimal value for power law fit
31.93026166420847%
"""

# show graph
plt.show()

quit()

edges_as_list = edg.values.tolist()
nodes_as_list = nod.values.tolist()
#print(edges_as_list) 

graph=draw_graph(edges_as_list, nodes_as_list)
# display graph
#nx.draw_networkx(graph)
