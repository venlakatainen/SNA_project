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

# read data

def get_twitter_data():
    # read edges
    edges = pd.read_csv("edges.csv")
    # read nodes
    nodes = pd.read_csv("nodes.csv")
    # return nodes and edges
    return edges, nodes

# draw graph 

def make_graph(edgelist):
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
    

def loglogplot(avg_deg):
    # average degree power law
    avg_deg_count = []
    
    # calculate how many node have same degree
    for l in range(len(avg_deg)):
        avg_deg_count.append(avg_deg[l][1])
    
    # draw figure as loglog plot
    avg_deg_counter = Counter(avg_deg_count)
    plt.figure(5)
    x_avg = np.array(list(avg_deg_counter.keys()))
    y_avg = np.array(list(avg_deg_counter.values()))
    plt.xscale("log")
    plt.yscale("log")
    
    plt.scatter(x_avg, y_avg, alpha=0.95)
    plt.plot(x_avg, y_avg, label="avg")
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.title('Avg degree log-log plot')
    
    

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


    
def plot_clustering_loglog_distribution(graph, num_bins=10):
    # compute the clustering coefficient for each node
    clustering = nx.clustering(graph)
    print(clustering)
    # create a histogram with the specified number of bins
    plt.hist(list(clustering.values()), bins=num_bins)
    plt.title("Power-Law Distribution of Clustering Coefficients")
    plt.xscale("log")
    plt.yscale("log")
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




def connected_sccs_distance(sccs, graph):
    
    # Compute the shortest path lengths between all pairs of nodes
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    # Build a set of all nodes in each SCC
    scc_nodes = [set(scc) for scc in sccs]

    # Build a set of all SCCs that are connected by a path of length up to 10
    connected_sccs = set()
    for i, scc1 in enumerate(sccs):
        for j, scc2 in enumerate(sccs):
            if i >= j:
                continue
            if any(shortest_paths.get(node1, {}).get(node2, float('inf')) <= 10 for node1 in scc1 for node2 in scc2):
                connected_sccs.add((i, j))

    # Build a new graph where each node represents a connected SCC
    group_graph = nx.DiGraph()
    for i, scc in enumerate(sccs):
        group_graph.add_node(i)

    # Add edges between SCCs that are connected
    for i, j in connected_sccs:
        group_graph.add_edge(i, j)

    return group_graph

            

# read csv
df = pd.read_csv('edges.csv', header=None, names=['Follower','Target'])


# select subset
df_sub = df.sample(10000, random_state=987)


# build graph from data frame
G = make_graph(df_sub)

# create graph from edges and plot
#nx.draw(G, with_labels=True, node_size=1000, alpha=0.5, arrows=True)
#plt.title('500 node subgraph')



# exercise 1
# calculate degrees 
in_deg, out_deg, avg_deg = get_degrees(G)
# plot degrees
#plot_degrees(in_deg, out_deg, avg_deg)


# exercise 2
# draw power-law figures
#draw_power_laws(in_deg, out_deg, avg_deg)


# exercise 3
#loglogplot(avg_deg)



# exercise 4
# get top 5 edge betweenness scores
#top_edges = top_edge_betweenness(G)
#print("Top 5 edges with highest betweenness centrality:")
#print(top_edges)

# get top 10 node betweenness scores
#top_nodes = top_node_betweenness(G)
#print("Top 10 nodes with highest betweenness centrality:")
#print(top_nodes)


# exercise 5
# clustering coefficient
plot_clustering_coefficient_distribution(G)

# exercise 6
# clustering coefficient power-law
plot_clustering_loglog_distribution(G)

# exercise 7
# strongly and weakly connected components
strong_components, weak_components = get_connected_components(G)

#print("Strongly connected components: ", strong_components)
#print("Weakly connected components: ", weak_components)
print("Number of strongly connected components: ", len(strong_components))
print("Number of weakly connected components: ", len(weak_components))


# exercise 8

# subgraph from strongly connected components
#CC = get_connected_components_graph(G, strong_components)


# draw the connected components graph
#pos = nx.spring_layout(CC)
#nx.draw(CC, pos, with_labels=True)
#nx.draw_networkx_edge_labels(CC, pos, edge_labels={(u,v):f"{u}->{v}" for u,v in CC.edges()})

# exercise 9
#distance_graph = connected_sccs_distance(list(strong_components), G)
#nx.draw(distance_graph)


# show graph
plt.show()

#quit()



