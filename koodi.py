import pandas as pd
import statistics
from scipy.stats import kurtosis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import powerlaw
import datetime
from collections import Counter

# read data

def get_twitter_data():
    # read edges
    edges = pd.read_csv("edges.csv")
    # read nodes
    nodes = pd.read_csv("nodes.csv")
    # return nodes and edges
    return edges, nodes


def make_graph(edgelist):
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
    ax1.set_ylabel("Number of Nodes")

    # out degree
    fig2 = plt.figure("Out degree", figsize=(8, 8))
    ax2 = fig2.add_subplot()
    degree_sequence = sorted((d for n, d in out_deg), reverse=True)
    ax2.bar(*np.unique(degree_sequence, return_counts=True), align='center', width=0.4)
    ax2.set_title("Out degree")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")

    # average degree
    fig3 = plt.figure("Average degree", figsize=(8, 8))
    ax3 = fig3.add_subplot()
    degree_sequence = sorted((d for n, d in avg_deg), reverse=True)
    ax3.bar(*np.unique(degree_sequence, return_counts=True), align='center', width=0.4)
    ax3.set_title("Average Degree")
    ax3.set_xlabel("Degree")
    ax3.set_ylabel("Number of Nodes")

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
    #print(in_deg_counter.keys())
    #print(in_deg_counter.values())
    # plot the figure
    plt.figure(5)
    #plt.scatter(degree, nodes)
    plt.loglog(in_deg_counter.keys(), in_deg_counter.values(), label="loglog")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('LogLog Distribution of in-degree') 

def top_edge_betweenness(graph):
    #Calculates the top five edges with the highest edge betweenness scores
    print("Edge betweenness")
    edge_betweenness = nx.edge_betweenness_centrality(graph)
    top_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_edges

def top_node_betweenness(graph):
    #Calculates the top ten nodes with the highest node betweenness scores
    print("Node betweenness")
    node_betweenness = nx.betweenness_centrality(graph)
    top_nodes = sorted(node_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_nodes

def plot_clustering_coefficient_distribution(graph, num_bins=10):
    # compute the clustering coefficient for each node
    clustering = nx.clustering(graph)

    # create a histogram with the specified number of bins
    plt.hist(list(clustering.values()), bins=num_bins)
    plt.title("Distribution of Clustering Coefficients")
    plt.xlabel("Clustering Coefficient")
    plt.ylabel("Number of Nodes")
    plt.show()

def get_connected_components(G):
    # identify the strongly connected components
    strong_components = list(nx.strongly_connected_components(G))

    # identify the weakly connected components
    weak_components = list(nx.weakly_connected_components(G))

    # return the results as a tuple
    return strong_components, weak_components
    
def get_connected_components_graph(graph, components):
    # create a new graph for the connected components
    CC = nx.DiGraph()

    # add the nodes and edges from the original graph to the new graph
    for component in components:
        CC.add_nodes_from(component)
        for u in component:
            for v in graph.successors(u):
                if v in component:
                    CC.add_edge(u, v)

    # return the connected components graph
    return CC


start = datetime.datetime.utcnow()

# read csv
df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])

# select subset
#df_sub = df.sample(1000, random_state=987)

# build graph from data frame
#G = make_graph(df_sub)
G = make_graph(df)

# get top 5 edge betweenness scores
#top_edges = top_edge_betweenness(G)
#print("Top 5 edges with highest betweenness centrality:")
#print(top_edges)

# get top 10 node betweenness scores
#top_nodes = top_node_betweenness(G)
#print("Top 10 nodes with highest betweenness centrality:")
#print(top_nodes)

# clustering coefficient
plot_clustering_coefficient_distribution(G)
quit()

# strongly and weakly connected components
strong_components, weak_components = get_connected_components(G)
print("Strongly connected components: ", strong_components)
print("Weakly connected components: ", weak_components)

# subgraph from strongly connected components
CC = get_connected_components_graph(strong_components, G)

# draw the connected components graph
pos = nx.spring_layout(CC)
nx.draw(CC, pos, with_labels=True)
nx.draw_networkx_edge_labels(CC, pos, edge_labels={(u,v):f"{u}->{v}" for u,v in CC.edges()})

# create graph from edges and plot
#nx.draw(G, with_labels=True, node_size=1000, alpha=0.5, arrows=True)
#plt.title('1000 node subgraph')

# in degree, out degree and average degree
#in_deg, out_deg, avg_deg = get_degrees(G)
#plot_degrees(in_deg, out_deg, avg_deg)

# draw power-law figures
#draw_power_laws(in_deg, out_deg, avg_deg)

# draw log-log
#loglogplot(in_deg)

#degree_sequence = sorted((d for n, d in out_deg), reverse=True)
#fit = powerlaw.Fit(degree_sequence)
"""
Calculating best minimal value for power law fit
31.93026166420847%
"""
print("Took: ", datetime.datetime.utcnow() - start)

# show graph
plt.show()
