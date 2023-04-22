import pandas as pd
import statistics
from scipy.stats import kurtosis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw
from collections import Counter

# read data

def get_twitter_data():
    # read edges
    edges = pd.read_csv("edges.csv")
    # read nodes
    nodes = pd.read_csv("nodes.csv")
    # return nodes and edges
    return edges, nodes

# draw graph 

def draw_graph(edgelist):
    # draw graph from given edgelist
    G = nx.from_pandas_edgelist(edgelist, 'Follower', 'Target', create_using=nx.DiGraph())
    # return created graph
    return G


# get degrees of given graph

def get_degrees(graph):
    # get in degree of given graph
    in_deg = list(graph.in_degree())
    # get out degree of given graph
    out_deg = list(graph.out_degree())
    
    avg_deg = []
    # get average degree of given graph
    for i in range(len(in_deg) - 1):
        avg_deg.append((in_deg[i][0], (in_deg[i][1] + out_deg[i][1]) / 2))
    
    # sort degree lists 
    in_deg.sort(key = lambda x:x[1])
    out_deg.sort(key = lambda x:x[1])
    avg_deg.sort(key = lambda x:x[1])
    
    return in_deg, out_deg, avg_deg


# plot degrees
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
    # in degree power law
    in_deg_count = []
    
    # calculate how many node have same degree
    for i in range(len(in_deg)):
        in_deg_count.append(in_deg[i][1])
        
    in_deg_counter = Counter(in_deg_count)
    
    # draw figure 
    plt.figure(2)
    x_in = in_deg_counter.keys()
    y_in = in_deg_counter.values()
    plt.plot(x_in, y_in, label="in")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('In degree power-law distribution')
    
    
    # out degree power law
    out_deg_count = []
    
    # calculate how many node have same degree
    for j in range(len(out_deg)):
        out_deg_count.append(out_deg[j][1])
        
    
    out_deg_counter = Counter(out_deg_count)
    
    # draw figure 
    plt.figure(3)
    x_out = out_deg_counter.keys()
    y_out = out_deg_counter.values()
    plt.plot(x_out, y_out, label="out")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('Out degree power-law distribution')
    
    # average degree power law
    avg_deg_count = []
    
    # calculate how many node have same degree
    for l in range(len(avg_deg)):
        avg_deg_count.append(avg_deg[l][1])
    
    # draw figure
    avg_deg_counter = Counter(avg_deg_count)
    plt.figure(4)
    x_avg = avg_deg_counter.keys()
    y_avg = avg_deg_counter.values()
    plt.plot(x_avg, y_avg, label="avg")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('Avg degree power-law distribution')
    


def loglogplot(in_deg):
    # calculate how many node have same degree
    in_deg_count = []
    nodes = []
    degree = []
   
    for i in range(len(in_deg)):
        in_deg_count.append(in_deg[i][1])
        nodes.append(in_deg[i][0])
        degree.append(in_deg[i][1])
        
    in_deg_counter = Counter(in_deg_count)
    print(in_deg_counter.keys())
    print(in_deg_counter.values())
    # plot the figure
    plt.figure(5)
    #plt.scatter(degree, nodes)
    plt.loglog(in_deg_counter.keys(), in_deg_counter.values(), label="loglog")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('LogLog Distribution of in-degree') 
        
        
        
# read csv
df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])


# select subset
df_sub = df.sample(500, random_state=987)


# build graph from data frame
G = draw_graph(df_sub)


# create graph from edges and plot
nx.draw(G, with_labels=True, node_size=1000, alpha=0.5, arrows=True)
plt.title('500 node subgraph')


# calculate degrees
in_deg, out_deg, avg_deg = get_degrees(G)
# plot degrees
#plot_degrees(in_deg, out_deg, avg_deg)

# draw power-law figures
draw_power_laws(in_deg, out_deg, avg_deg)

loglogplot(in_deg)

#degree_sequence = sorted((d for n, d in out_deg), reverse=True)
#fit = powerlaw.Fit(degree_sequence)
"""
Calculating best minimal value for power law fit
31.93026166420847%
"""

# show graph
plt.show()

quit()

#edges_as_list = edg.values.tolist()
#nodes_as_list = nod.values.tolist()
#print(edges_as_list) 

#graph=draw_graph(edges_as_list, nodes_as_list)
# display graph
#nx.draw_networkx(graph)