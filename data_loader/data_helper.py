import networkx as nx
from sklearn import preprocessing
import numpy as np
import os
import collections
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import permutations, combinations
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from numpy.linalg import matrix_power
from scipy import sparse
import pickle
import copy

tf.disable_eager_execution()

NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38, 'PROTEINS': 3, 'PTC': 22, 'DD': 89}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize(graph):
    D_inv = np.diag(np.sum(graph, axis=0) ** -0.5)
    graph = np.matmul(np.matmul(D_inv, graph), D_inv)
    return graph

def A_power(graph_adj):
  top = graph_adj.shape[0]-1
  D_inv = np.diag(np.sum(graph_adj, axis=0) ** -0.5)
  graph_adj = np.matmul(np.matmul(D_inv, graph_adj), D_inv)
  adj_powers=[matrix_power(graph_adj,i+1) - matrix_power(graph_adj,i) for i in range(1, top+1)]
  adj_powers.insert(0,graph_adj)
  return np.array(adj_powers)
  #top = graph_adj.shape[0]
  #adj_powers, diffs = [],[]
  #adj_powers.append(graph_adj)
  #diffs.append(graph_adj)
  #for p in range(2,top+1):
  #   power, diff = correct_A_power(p, graph_adj, adj_powers)
  #   adj_powers.append(power), diffs.append(diff)
  return np.array(diffs)

def correct_A_power(p,graph_adj,adj_powers):
    adj_power = matrix_power(graph_adj,p) + adj_powers[-1]
    np.fill_diagonal(adj_power, 0)
    adj_power[np.where(adj_power > 0)] = 1
    diff = adj_power - adj_powers[-1]
    return adj_power, diff




def load_dataset_ori(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name]+1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1] = 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
           # curr_graph = noramlize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2,0,1])


    return graphs, np.array(labels)

def load_dataset(ds_name):
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
            labels.append(int(graph_meta[1]))  # ori
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0]) + 1] = int(vertex[0]) + 1
                for k in range(2, len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            # curr_graph = noramlize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs)
    labels = np.array(labels)  # ori

#    dim = [graph.shape[0] for graph in graphs]
#    sort = (sorted([(x, i) for (i, x) in enumerate(dim)], reverse=True)[:110])
#    graphs = np.delete(graphs, ([sort[i][1] for i in range(len(sort))]), axis=0)
#    labels = np.delete(labels, ([sort[i][1] for i in range(len(sort))]), axis=0)

    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])  ## ori: use all features
    #    edge_feature = Edge_Label(graphs[i])
    #    adj_powers = A_power(graphs[i][0])
    #    graphs[i] = np.concatenate((adj_powers, edge_feature), axis=0)
        adj_powers = A_power(graphs[i][0])
        graphs[i] = np.concatenate((graphs[i], adj_powers[1:]), axis=0)

   # max_dim = max([graph.shape[0] for graph in graphs])
   # for i in range(graphs.shape[0]):
   #     padded = np.zeros((max_dim - graphs[i].shape[0], graphs[i].shape[1], graphs[i].shape[2]))
   #     graphs[i] = np.concatenate((graphs[i], padded), axis=0)

    return graphs, labels

def load_dataset2s(ds_name):

    graph_dict=dict(zip([5,6,9,12,15,16,25], [0.7,0.7,0.6,0.8,0.8,0.8,0.7]))
    num_rep=[100,100,100,200,200,200,200]
  #  graph_dict=dict(zip([5,6,9,12,15,16], [0.7,0.7,0.6,0.8, 0.8,0.8]))
  #  num_rep=[100,100,100,200,200,200]
    graphs = []
    labels = []
    for  num, (k,v) in zip(num_rep, graph_dict.items()):
      G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
      #plt.subplot(121)
      #nx.draw(G,with_labels=True)
      label=nx.clique.graph_clique_number(G)
      A=nx.to_numpy_matrix(G,nodelist=list(range(len(G.nodes))))
      graphs.append(A)
      labels.append(label)
      for graph in range(num):
        node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
        G_new = nx.relabel_nodes(G, node_mapping)
        A_new=nx.to_numpy_matrix(G_new, nodelist=list(range(len(G_new.nodes))))
        graphs.append(A_new)
        labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)
    for i in range(graphs.shape[0]):
    #    graphs[i] = A_power(graphs[i])
         graphs[i] = np.expand_dims(graphs[i], axis=0)  # use only A

#    max_dim = max([graph.shape[0] for graph in graphs])
#    for i in range(graphs.shape[0]):
#        padded = np.zeros((max_dim-graphs[i].shape[0], graphs[i].shape[1], graphs[i].shape[1]))
#        graphs[i] =np.concatenate([graphs[i], padded], axis=0)

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    return graphs, labels


def load_dataset_2s_val(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """

    graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
    num_rep = [20, 20, 20, 30, 30, 30, 30]
    # graph_dict=dict(zip([5,6,9], [0.6,0.7,0.6]))
    # num_rep=[3,3,3]
    graphs = []
    labels = []
    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
        # plt.subplot(121)
        # nx.draw(G,with_labels=True)
        label = nx.clique.graph_clique_number(G)
        A = nx.to_numpy_matrix(G, nodelist=list(range(len(G.nodes))))
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            u, v = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(u, v)
            if G_new.number_of_edges() == G.number_of_edges() + 1:
                if nx.clique.graph_clique_number(G_new) == label:
                    A_new = nx.to_numpy_matrix(G_new, nodelist=list(range(len(G_new.nodes))))
                    graphs.append(A_new)
                    labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)
    for i in range(graphs.shape[0]):
        #    graphs[i] = np.transpose(graphs[i], [2,0,1])         ## ori: use all features
        graphs[i] = np.expand_dims(np.expand_dims(graphs[i], axis=0), axis=0)  # use only A

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    return graphs, labels


def load_dataset2m(ds_name):
    graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
    num_rep = [100, 100, 100, 200, 200, 200, 200]
    # graph_dict=dict(zip([5,6,9], [0.6,0.7,0.6]))
    # num_rep=[3,3,3]
    graphs = []
    labels = []
    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
        # plt.subplot(121)
        # nx.draw(G,with_labels=True)
        label = nx.clique.graph_clique_number(G)
        A = nx.to_numpy_matrix(G, nodelist=list(range(len(G.nodes))))
        graphs.append(A)
        labels.append(label)
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            u, v = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(u, v)
            if G_new.number_of_edges() == G.number_of_edges() + 1:
                if nx.clique.graph_clique_number(G_new) == label:
                    A_new = nx.to_numpy_matrix(G_new, nodelist=list(range(len(G_new.nodes))))
                    graphs.append(A_new)
                    labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)
    for i in range(graphs.shape[0]):
        #    graphs[i] = np.transpose(graphs[i], [2,0,1])         ## ori: use all features
        graphs[i] = np.expand_dims(graphs[i], axis=0)  # use only A

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    #  idx = np.where(labels == 4)[0]          #  balance data
    #  labels = np.delete(labels, idx[:700])  #  labels = labels[:2000]
    #  graphs = np.delete(graphs, idx[:700], axis=0)       #  graphs= graphs[:2000]
    return graphs, labels


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels, graphs3d):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]
    graphs3d = graphs3d[indexes]
    r_graphs = []
    r_labels = []
    r_graphs3d = []
    one_size = []
    one_size_node = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
            one_size_node.append(np.expand_dims(graphs3d[i], axis=0))

        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_graphs3d.append(np.concatenate(one_size_node, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            one_size_node = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
            one_size_node.append(np.expand_dims(graphs3d[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_graphs3d.append(np.concatenate(one_size_node, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels, r_graphs3d

def QM9_group_same_size(graphs1d, graphs2d, graphs3d, labels):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs2d))
    indexes = np.argsort(sizes)
    graphs1d = graphs1d[indexes]
    graphs2d = graphs2d[indexes]
    graphs3d = graphs3d[indexes]
    labels = labels[indexes]
    r_graphs1d, r_graphs2d ,r_graphs3d = [], [], []
    r_labels = []
    one_size1d, one_size2d, one_size3d = [],[],[]
    start = 0
    size = graphs2d[0].shape[-1]
    for i in range(len(graphs2d)):
        if graphs2d[i].shape[-1] == size:
            one_size1d.append(np.expand_dims(graphs1d[i], axis=0))
            one_size2d.append(np.expand_dims(graphs2d[i], axis=0))
            one_size3d.append(np.expand_dims(graphs3d[i], axis=0))

        else:
            r_graphs1d.append(np.concatenate(one_size1d, axis=0))
            r_graphs2d.append(np.concatenate(one_size2d, axis=0))
            r_graphs3d.append(np.concatenate(one_size3d, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size1d ,one_size2d ,one_size3d = [], [], []
            size = graphs2d[i].shape[-1]
            one_size1d.append(np.expand_dims(graphs1d[i], axis=0))
            one_size2d.append(np.expand_dims(graphs2d[i], axis=0))
            one_size3d.append(np.expand_dims(graphs3d[i], axis=0))
    r_graphs1d.append(np.concatenate(one_size1d, axis=0))
    r_graphs2d.append(np.concatenate(one_size2d, axis=0))
    r_graphs3d.append(np.concatenate(one_size3d, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs1d, r_graphs2d, r_graphs3d, r_labels

# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels, graphs3d):
    r_graphs, r_labels, r_graphs3d = [], [], []
    for i in range(len(labels)):
        curr_graph, curr_labels, curr_nodefeature = shuffle(graphs[i], labels[i], graphs3d[i])
        r_graphs.append(curr_graph)
        r_graphs3d.append(curr_nodefeature )
        r_labels.append(curr_labels)
    return r_graphs, r_labels, r_graphs3d


def QM9_shuffle_same_size(graphs1d, graphs2d, graphs3d, labels):
    r_graphs1d, r_graphs2d, r_labels, r_graphs3d = [], [], [], []
    for i in range(len(labels)):
        curr_graph1d, curr_graph2d,curr_graph3d, curr_labels = QM9_shuffle(graphs1d[i], graphs2d[i], graphs3d[i],labels[i])
        r_graphs1d.append(curr_graph1d)
        r_graphs2d.append(curr_graph2d)
        r_graphs3d.append(curr_graph3d )
        r_labels.append(curr_labels)
    return r_graphs1d, r_graphs2d, r_graphs3d, r_labels



def split_to_batches(graphs, labels, graphs3d, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs = []
    r_labels = []
    r_graphs3d = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_graphs3d = r_graphs3d + np.split(graphs3d[k], [j for j in range(size, graphs3d[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
    return np.array(r_graphs), np.array(r_labels), np.array(r_graphs3d)

def QM9_split_to_batches(graphs1d, graphs2d, graphs3d, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs1d, r_graphs2d, r_graphs3d  = [],[],[]
    r_labels = []
    for k in range(len(graphs2d)):
        r_graphs1d = r_graphs1d + np.split(graphs1d[k], [j for j in range(size, graphs1d[k].shape[0], size)])
        r_graphs2d = r_graphs2d + np.split(graphs2d[k], [j for j in range(size, graphs2d[k].shape[0], size)])
        r_graphs3d = r_graphs3d + np.split(graphs3d[k], [j for j in range(size, graphs3d[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
    return np.array(r_graphs1d), np.array(r_graphs2d), np.array(r_graphs3d), np.array(r_labels)

# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels, graphs3d):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    #np.random.seed(1)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf], np.array(graphs3d)[shf]


def QM9_shuffle(graphs1d,graphs2d,graphs3d, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    #np.random.seed(1)
    np.random.shuffle(shf)
    return np.array(graphs1d)[shf],np.array(graphs2d)[shf] , np.array(graphs3d)[shf],  labels[shf]


def noramlize_graph(curr_graph):
    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg != 0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels = np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)


def load_dataset3s(ds_name, upper=True):
    graphs, adj_powers, graphs3d = [], [], []
    labels = []
    if ds_name == 'syn':
        graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
        #   graph_dict=dict(zip([5,6,9,12], [0.7,0.7,0.6,0.8,0.8,0.8,0.7]))
        num_rep = [100, 100, 100, 200, 200, 200, 200]
        for num, (k, v) in zip(num_rep, graph_dict.items()):
            G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
            adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
            graphs.append(adj)
            label = nx.clique.graph_clique_number(G)
            if upper == False:
                A = construct_A3(G)
                adj_power = A_power(adj)
            else:
                A = construct_upperA3(G)
                adj_power = A_power(adj)
            graphs3d.append(A)
            adj_powers.append(adj_power)
            labels.append(label)
            for graph in range(num):
                node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
                G_new = nx.relabel_nodes(G, node_mapping)
                adj_new = nx.linalg.graphmatrix.adjacency_matrix(G_new).toarray()
                if upper == False:
                    A_new = construct_A3(G_new)
                    adj_power = A_power(adj_new)
                else:
                    A_new = construct_upperA3(G_new)
                    adj_power = A_power(adj_new)
                graphs.append(adj)
                graphs3d.append(A_new)
                adj_powers.append(adj_power)
                labels.append(label)
            if k == list(graph_dict.keys())[-1]:
                    zero = np.zeros((k + 1, k + 1, k + 1))
                    graphs.append(zero)
                    adj_powers.append(zero)
                    graphs3d.append(zero)

        graphs = np.array(graphs)
        labels = np.array(labels)
        graphs3d = np.array(graphs3d)
        adj_powers = np.array(adj_powers)

        for i in range(graphs.shape[0]):
        #    graphs[i] = np.expand_dims(graphs[i], axis=0)
            graphs[i] =  adj_powers[i]
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)

        graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
        graphs3d = tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())

        graphs = np.delete(graphs, -1, axis=0)
        graphs3d = np.delete(graphs3d, -1, axis=0)

        #  graphs = np.delete(graphs, -1, axis=0)
        le = preprocessing.LabelEncoder()  # to find clique
        le.fit(labels)  # to find clique
        labels = le.transform(labels)  # to find clique

    else:
        directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
        with open(directory, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
                labels.append(int(graph_meta[1]))  # ori
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    if NUM_LABELS[ds_name] != 0:
                        curr_graph[j, j, int(vertex[0]) + 1] = 1.
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k]), 0] = 1.
                #      curr_graph = noramlize_graph(curr_graph)
                graphs.append(curr_graph)
        graphs = np.array(graphs)
        for i in range(graphs.shape[0]):
            graphs[i] = np.expand_dims(np.transpose(graphs[i], [2, 0, 1])[0], axis=0)  # use only A
            G = nx.from_numpy_array(graphs[i][0])
            graphs[i] = construct_upperA3(G)
            graphs[i] = np.expand_dims(graphs[i], axis=0)

        labels = np.array(labels)


    return graphs, labels, graphs3d


def load_dataset_3s_val(ds_name, upper):
    graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
    # graph_dict=dict(zip([5,6,9,12], [0.7,0.7,0.6,0.8,0.8,0.8,0.7]))
    num_rep = [20, 20, 20, 30, 30, 30, 30]
    # graph_dict=dict(zip([5,6,9], [0.6,0.7,0.6]))
    # num_rep=[3,3,3]
    graphs, adj_powers, graphs3d = [], [], []
    labels = []
    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
        label = nx.clique.graph_clique_number(G)
        if upper == False:
            A = construct_A3(G)
        else:
            A = construct_upperA3(G)
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            u, v = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(u, v)
            if G_new.number_of_edges() == G.number_of_edges() + 1:
                if nx.clique.graph_clique_number(G_new) == label:
                    adj = nx.linalg.graphmatrix.adjacency_matrix(G_new).toarray()
                    if upper == False:
                        A_new = construct_A3(G_new)
                        adj_power = A_power(adj)
                    else:
                        A_new = construct_upperA3(G_new)
                        adj_power = A_power(adj)
                    graphs3d.append(A_new)
                    labels.append(label)
                    graphs.append(adj_power)
    graphs = np.array(graphs)
    labels = np.array(labels)
    graphs3d = np.array(graphs3d)

   # graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
    for i in range(graphs.shape[0]):
        graphs[i] = np.expand_dims(graphs[i], axis=0)
        graphs3d[i] = np.expand_dims(np.expand_dims(graphs3d[i], axis=0), axis=0)

    graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
    graphs3d = tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())

  #  for i in range(graphs.shape[0]):
  #      graphs[i] = np.expand_dims(np.expand_dims(graphs[i], axis=0), axis=0)
  #      graphs3d[i] = np.expand_dims( np.expand_dims(graphs3d[i], axis=0), axis=0)

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    return graphs, labels, graphs3d


def load_dataset3m(ds_name, upper):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution
    graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
    num_rep = [100, 100, 100, 200, 200, 200, 200]
    # graph_dict=dict(zip([5,6,9], [0.6,0.7,0.6]))
    # num_rep=[3,3,3]
    graphs = []
    labels = []

    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G = nx.erdos_renyi_graph(k, v, seed=1, directed=False)
        label = nx.clique.graph_clique_number(G)
        if upper == False:
            A = construct_A3(G)
        else:
            A = construct_upperA3(G)
        graphs.append(A)
        labels.append(label)
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            u, v = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(u, v)
            if G_new.number_of_edges() == G.number_of_edges() + 1:
                if nx.clique.graph_clique_number(G_new) == label:
                    if upper == False:
                        A_new = construct_A3(G_new)
                    else:
                        A_new = construct_upperA3(G_new)
                    graphs.append(A_new)
                    labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)

    graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    #  idx = np.where(labels == 4)[0]          #  balance data
    #  labels = np.delete(labels, idx[:700])  #  labels = labels[:2000]
    #  graphs = np.delete(graphs, idx[:700], axis=0)       #  graphs= graphs[:2000]
    return graphs, labels


def load_dataset3s_large(ds_name, upper):
    graph_dict = dict(zip([7, 8, 9], [1, 1, 1, 1, 1, 1, 1]))
    num_rep = [20, 20, 20, 50, 50, 200, 200]

    graphs = []
    labels = []
    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G, label = construct_graph(k, v, sub_size=1)
        if upper == False:
            A = construct_A3(G)
        else:
            A = construct_upperA3(G)
        graphs.append(A)
        labels.append(label)
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            if upper == False:
                A_new = construct_A3(G_new)
            else:
                A_new = construct_upperA3(G_new)
            graphs.append(A_new)
            labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)
    max_dim = max([graph.shape[0] for graph in graphs]) + 1

    for i in range(graphs.shape[0]):
        padded = np.zeros((max_dim, max_dim, max_dim))
        padded[:graphs[i].shape[0], :graphs[i].shape[1], :graphs[i].shape[2]] = graphs[i]
        graphs[i] = padded

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    return graphs, labels


def load_dataset_3s_large_val(ds_name, upper):
    graph_dict = dict(zip([7, 8, 9], [1, 1, 1, 1, 1, 1, 1]))
    num_rep = [15, 15, 15, 50, 50, 200, 200]

    graphs = []
    labels = []
    for num, (k, v) in zip(num_rep, graph_dict.items()):
        G, label = construct_graph(k, v, sub_size=1)
        for graph in range(num):
            node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            G_new = nx.relabel_nodes(G, node_mapping)
            f, t = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(f, t)
            f, t = random.sample(range(G_new.number_of_nodes() + 1), 2)
            G_new.add_edge(f, t)
            if G_new.number_of_edges() >= G.number_of_edges() + 1:
                if upper == False:
                    A_new = construct_A3(G_new)
                else:
                    A_new = construct_upperA3(G_new)
                graphs.append(A_new)
                labels.append(label)
    graphs = np.array(graphs)
    labels = np.array(labels)
    max_dim = max([graph.shape[0] for graph in graphs])

    for i in range(graphs.shape[0]):
        padded = np.zeros((max_dim, max_dim, max_dim))
        padded[:graphs[i].shape[0], :graphs[i].shape[1], :graphs[i].shape[2]] = graphs[i]
        graphs[i] = padded

    graphs = list(graphs)
    for i in range(len(graphs)):
        #    graphs[i] = np.transpose(graphs[i], [2,0,1])         ## ori: use all features
        graphs[i] = np.expand_dims(graphs[i], axis=0)

    le = preprocessing.LabelEncoder()  # to find clique
    le.fit(labels)  # to find clique
    labels = le.transform(labels)  # to find clique
    return graphs, labels


def construct_graph(k, v, sub_size):
    G = nx.erdos_renyi_graph(k, v, directed=False)
    sub_k, sub_v = np.int(k * sub_size), 0.1
    G2 = nx.erdos_renyi_graph(sub_k, sub_v, directed=False)
    G3 = nx.disjoint_union(G, G2)
    G3.add_edge(G.number_of_nodes() - 1, G.number_of_nodes())
    label = nx.clique.graph_clique_number(G3)
    return G3, label


def get_cliques_by_length(G, length_clique):
    """ Return the list of all cliques in an undirected graph G with length
    equal to length_clique. """
    cliques = []
    for c in nx.enumerate_all_cliques(G):
        if len(c) <= length_clique:
            if len(c) == length_clique:
                cliques.append(c)
        else:
            return cliques
    # return empty list if nothing is found
    return cliques


def construct_A3(G, length_clique=3):
    tri = get_cliques_by_length(G, 3)
    # print(tri)
    nn = G.number_of_nodes()
    A3 = np.zeros((nn, nn, nn), dtype='float32')
    for i in tri:
        perm = permutations(i)
        for j in list(perm):
            A3[j] = 1
    return A3


def construct_upperA3(G, length_clique=3):
    tri = get_cliques_by_length(G, 3)
    # print(tri)
    nn = G.number_of_nodes()
    A3 = np.zeros((nn, nn, nn), dtype='float32')
    for i in tri:
        A3[tuple(i)] = 1
    return A3


def motif(shape):
    target = nx.Graph()
    if shape == 'tree':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
    if shape == 'triangle':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(1, 3)
    if shape == 'tail_triangle':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(1, 3)
        target.add_edge(1, 4)
    if shape == 'star':
        target.add_edge(1, 2)
        target.add_edge(1, 3)
        target.add_edge(1, 4)
    if shape == 'chain':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
    if shape == 'box':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
    if shape == 'semi_clique':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
        target.add_edge(1, 3)
    if shape == '4_clique':
        target.add_edge(1, 2)
        target.add_edge(2, 3)
        target.add_edge(3, 4)
        target.add_edge(1, 4)
        target.add_edge(1, 3)
        target.add_edge(2, 4)

    return target


def high_order(g, target):
    nn = g.number_of_nodes()
    sub_node = []
    if target.number_of_nodes() == 3:
        A = np.zeros((nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                A[tuple(subg.nodes())] = 1
                sub_node.append(tuple(subg.nodes()))
    if target.number_of_nodes() == 4:
        A = np.zeros((nn, nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                A[tuple(subg.nodes())] = 1
                sub_node.append(tuple(subg.nodes()))

    label = len(sub_node)

    return A, label, sub_node

def high_order2(g, target):
    nn = g.number_of_nodes()
    sub_node = []
    if target.number_of_nodes() == 3:
        A = np.zeros((nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                center_node = list(set(list(subg.edges)[0]).intersection(set(list(subg.edges)[1])))
                edge_nodes = list(set(tuple(subg.nodes())).difference(set((center_node))))
                A[center_node[0], edge_nodes[0], edge_nodes[1]] = 1
                A[center_node[0], edge_nodes[1], edge_nodes[0]] = 1
                A[edge_nodes[0], center_node[0], edge_nodes[1]] = 1
                A[edge_nodes[1], center_node[0], edge_nodes[0]] = 1

                sub_node.append(tuple(subg.nodes()))
    if target.number_of_nodes() == 4:
        A = np.zeros((nn, nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                A[tuple(subg.nodes())] = 1
                sub_node.append(tuple(subg.nodes()))

    label = len(sub_node)

    return A, label, sub_node

def high_order3(g, target):
    nn = g.number_of_nodes()
    sub_node = []
    if target.number_of_nodes() == 3:
        A1, A2 = np.zeros((nn, nn, nn), dtype='float32'), np.zeros((nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                center_node = list(set(list(subg.edges)[0]).intersection(set(list(subg.edges)[1])))
                edge_nodes = list(set(tuple(subg.nodes())).difference(set((center_node))))
                A1[center_node[0], edge_nodes[0], edge_nodes[1]] = 1
                A1[center_node[0], edge_nodes[1], edge_nodes[0]] = 1
                A2[edge_nodes[0], center_node[0], edge_nodes[1]] = 2
                A2[edge_nodes[1], center_node[0], edge_nodes[0]] = 2

                sub_node.append(tuple(subg.nodes()))

    if target.number_of_nodes() == 4:
        A = np.zeros((nn, nn, nn, nn), dtype='float32')
        for sub_nodes in combinations(g.nodes(), len(target.nodes())):
            subg = g.subgraph(sub_nodes)
            if nx.is_connected(subg) and nx.is_isomorphic(subg, target):
                A[tuple(subg.nodes())] = 1
                sub_node.append(tuple(subg.nodes()))

    label = len(sub_node)

    return A1, A2, label, sub_node


def multihead(ds_name, target_shape):
    graphs,  graphs3d, labels = [], [], []
    if ds_name == 'syn':
        target = motif(target_shape)

      #  graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
      #  num_rep = [100, 100, 100, 200, 200, 200, 200]
        graph_dict = dict(zip([8, 9, 9, 10, 10, 11, 11, 12, 13], [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.2, 0.2]))
        num_rep = [50, 50, 50, 50, 100, 100, 100, 100, 100, 100]

        for num, (k, v) in zip(num_rep, graph_dict.items()):
             for s in range(num):
                G = nx.erdos_renyi_graph(k, v, seed=s, directed=False)
                if nx.is_connected(G):
                    graph3d, label, _ = high_order(G, target)
                   # label = nx.clique.graph_clique_number(G)
                    labels.append(label)
                    graphs3d.append(graph3d)
                    adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
                    graphs.append(adj)
        graphs = np.array(graphs)
        graphs3d = np.array(graphs3d)
        for i in range(graphs.shape[0]):
            graphs[i] = np.expand_dims(graphs[i], axis=0)
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
 #       le = preprocessing.LabelEncoder()  # to find clique
 #       le.fit(labels)  # to find clique
 #       labels = le.transform(labels)  # to find clique

    else:
        target = motif(target_shape)
        directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)

        with open(directory, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
                labels.append(int(graph_meta[1]))  # ori
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    if NUM_LABELS[ds_name] != 0:
                        curr_graph[j, j, int(vertex[0]) + 1] = 1.
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k]), 0] = 1.
                #      curr_graph = noramlize_graph(curr_graph)
                graphs.append(curr_graph)
        graphs = np.array(graphs)
        labels = np.array(labels)
    #    dim = [graph.shape[0] for graph in graphs]
    #    sort = (sorted([(x, i) for (i, x) in enumerate(dim)], reverse=True)[:100])
    #    graphs = np.delete(graphs, ([sort[i][1] for i in range(len(sort))]), axis=0)
    #    labels = np.delete(labels, ([sort[i][1] for i in range(len(sort))]), axis=0)

        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][0])
            graph3d, _, _ = high_order(G, target)
            graphs3d.append(graph3d)
            adj_powers = A_power(graphs[i][0])
            graphs[i] = np.concatenate((graphs[i], adj_powers[1:]), axis=0)
        graphs3d = np.array(graphs3d)
        for i in range(graphs3d.shape[0]):
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
    return graphs, np.array(labels), graphs3d

def gnn3(ds_name, target_shape):
    graphs,  graphs3d, labels, adj_powers =[], [], [], []
    if ds_name == 'syn':
        target = motif(target_shape)

      #  graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
      #  num_rep = [100, 100, 100, 200, 200, 200, 200]
        graph_dict = dict(zip([8, 9, 9, 10, 10, 11, 11, 12, 13], [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.2, 0.2]))
        num_rep = [50, 50, 50, 50, 100, 100, 100, 100, 100, 100]

        for num, (k, v) in zip(num_rep, graph_dict.items()):
             for s in range(num):
                G = nx.erdos_renyi_graph(k, v, seed=s, directed=False)
                if nx.is_connected(G):
                    graph3d, label, _ = high_order(G, target)
                   # label = nx.clique.graph_clique_number(G)
                    labels.append(label)
                    graphs3d.append(graph3d)
                    adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
                    graphs.append(adj)
        graphs = np.array(graphs)
        graphs3d = np.array(graphs3d)
        for i in range(graphs.shape[0]):
            graphs[i] = np.expand_dims(graphs[i], axis=0)
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
 #       le = preprocessing.LabelEncoder()  # to find clique
 #       le.fit(labels)  # to find clique
 #       labels = le.transform(labels)  # to find clique

    else:
        target = motif(target_shape)
        directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)

        with open(directory, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
                labels.append(int(graph_meta[1]))  # ori
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    if NUM_LABELS[ds_name] != 0:
                        curr_graph[j, j, int(vertex[0]) + 1] = 1.
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k]), 0] = 1.
                curr_graph = noramlize_graph(curr_graph)
                graphs.append(curr_graph)
        graphs = np.array(graphs)
        labels = np.array(labels)
       # dim = [graph.shape[0] for graph in graphs]
       # sort = (sorted([(x, i) for (i, x) in enumerate(dim)], reverse=True)[:100])
       # graphs = np.delete(graphs, ([sort[i][1] for i in range(len(sort))]), axis=0)
       # labels = np.delete(labels, ([sort[i][1] for i in range(len(sort))]), axis=0)

        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][0])
            graph3d, _, _ = high_order(G, target)
            adj_power = A_power(graphs[i][0])
            graphs3d.append(graph3d)
            adj_powers.append(adj_power)

        graphs3d = np.array(graphs3d)
        adj_powers = np.array(adj_powers)
        for i in range(graphs3d.shape[0]):
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
            adj_powers[i] = np.expand_dims(adj_powers[i], axis=0)
            graphs3d[i] = np.concatenate((graphs3d[i], adj_powers[i]), axis=0)
     #   graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
     #   graphs3d = tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())

    return graphs, np.array(labels), graphs3d

def gnn4(ds_name, target_shape):
    graphs,  graphs3d, labels, adj_powers =[], [], [], []
    if ds_name == 'syn':
        target = motif(target_shape)

      #  graph_dict = dict(zip([5, 6, 9, 12, 15, 16, 25], [0.7, 0.7, 0.6, 0.8, 0.8, 0.8, 0.7]))
      #  num_rep = [100, 100, 100, 200, 200, 200, 200]
        graph_dict = dict(zip([8, 9, 9, 10, 10, 11, 11, 12, 13], [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.2, 0.2]))
        num_rep = [50, 50, 50, 50, 100, 100, 100, 100, 100, 100]

        for num, (k, v) in zip(num_rep, graph_dict.items()):
             for s in range(num):
                G = nx.erdos_renyi_graph(k, v, seed=s, directed=False)
                if nx.is_connected(G):
                    graph3d, label, _ = high_order(G, target)
                   # label = nx.clique.graph_clique_number(G)
                    labels.append(label)
                    graphs3d.append(graph3d)
                    adj = nx.linalg.graphmatrix.adjacency_matrix(G).toarray()
                    graphs.append(adj)
        graphs = np.array(graphs)
        graphs3d = np.array(graphs3d)
        for i in range(graphs.shape[0]):
            graphs[i] = np.expand_dims(graphs[i], axis=0)
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
 #       le = preprocessing.LabelEncoder()  # to find clique
 #       le.fit(labels)  # to find clique
 #       labels = le.transform(labels)  # to find clique

    else:
        target = motif(target_shape)
        directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)

        with open(directory, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
                labels.append(int(graph_meta[1]))  # ori
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    if NUM_LABELS[ds_name] != 0:
                        curr_graph[j, j, int(vertex[0]) + 1] = 1.
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k]), 0] = 1.
                #curr_graph = noramlize_graph(curr_graph)
                graphs.append(curr_graph)
        graphs = np.array(graphs)
        labels = np.array(labels)
        dim = [graph.shape[0] for graph in graphs]
        sort = (sorted([(x, i) for (i, x) in enumerate(dim)], reverse=True)[:100])
        graphs = np.delete(graphs, ([sort[i][1] for i in range(len(sort))]), axis=0)
        labels = np.delete(labels, ([sort[i][1] for i in range(len(sort))]), axis=0)

        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][0])
            graph3d, _, _ = high_order(G, target)
            adj_power = A_power(graphs[i][0])
            graphs3d.append(graph3d)
            adj_powers.append(adj_power)

        graphs3d = np.array(graphs3d)
        adj_powers = np.array(adj_powers)
        for i in range(graphs3d.shape[0]):
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
            adj_powers[i] = np.expand_dims(adj_powers[i], axis=0)
            graphs3d[i] = np.concatenate((graphs3d[i], adj_powers[i]), axis=0)
            graphs[i] = np.einsum('ijj->ij', graphs[i][1:])
     #   graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
     #   graphs3d = tf.ragged.constant(graphs3d).to_tensor().eval(session=tf.Session())

    return graphs, np.array(labels), graphs3d

def load_qm9_aux(which_set, target_param,target_shape):

    target = motif(target_shape)
    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)
    graphs, graphs1d, graphs2d,  graphs3d, labels, adj_powers =[], [], [], [],[], []
    counter = 0
    with open(base_path, 'rb') as f:
        data = pickle.load(f)

        for instance in data:
            counter += 1
            if counter == 100:
                break

            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 19))
            for i in range(13):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 13] = instance['usable_features']['distance_mat']
            graph[:, :, 14] = instance['usable_features']['affinity']
            graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
            graphs.append(graph)
    graphs = np.array(graphs)
    graphs_copy = copy.deepcopy(graphs)
    labels = np.array(labels).squeeze()  # shape N x 12
  #  if target_param is not False:  # regression over a specific target, not all 12 elements
  #      labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1


    for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][14])
            graph3d, _, _ = high_order2(G, target)
            adj_power = A_power(graphs[i][14])
            graphs3d.append(graph3d)
            adj_powers.append(adj_power)
            graph1d = graphs[i][:13]
            graph1d = np.einsum('ijj->ij', graph1d)
            graphs_copy[i] = graph1d
           # graphs[i] = graphs[i]
            graphs[i] = graphs[i][13:]
           # graphs[i][0] = normalize(graphs[i][0])
           # graphs[i][1] = normalize(graphs[i][1])

    graphs3d = np.array(graphs3d)
    adj_powers = np.array(adj_powers)
    for i in range(graphs3d.shape[0]):
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
            adj_powers[i] = np.expand_dims(adj_powers[i], axis=0)
            graphs3d[i] = np.concatenate((graphs3d[i], adj_powers[i]), axis=0)

    return graphs_copy, graphs, graphs3d, labels

def load_qm9(target_param,target_shape):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12, (or Nx1 is target_param is not False)
                 val_graphs: N_val,
                 val_labels: N_train x 12, (or Nx1 is target_param is not False)
                 test_graphs: N_test,
                 test_labels: N_test x 12, (or Nx1 is target_param is not False)
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    train_graphs1d, train_graphs2d, train_graphs3d, train_labels = load_qm9_aux('train', target_param,target_shape)
    val_graphs1d, val_graphs2d, val_graphs3d, val_labels = load_qm9_aux('val', target_param,target_shape)
    test_graphs1d, test_graphs2d, test_graphs3d, test_labels = load_qm9_aux('test', target_param,target_shape)
    return train_graphs1d, train_graphs2d, train_graphs3d, train_labels, val_graphs1d, val_graphs2d, val_graphs3d, val_labels, test_graphs1d, test_graphs2d, test_graphs3d, test_labels

def load_qm9_aux_gnn3(which_set,  target_param, target_shape):

    target = motif(target_shape)
    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)
    graphs, graphs3d, graphs3d2,labels, adj_powers =[], [], [], [],[]
    counter = 0
    with open(base_path, 'rb') as f:
        data = pickle.load(f)

        for instance in data:
            #counter += 1
            #if counter == 10000:
            #    break

            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 19))
            for i in range(13):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 13] = instance['usable_features']['distance_mat']
            graph[:, :, 14] = instance['usable_features']['affinity']
            graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
            #for i in range(4):
            #    graph[:,:,i] += graph[:,:,i+15]
            #graphs.append(graph[:,:,:15])
    graphs = np.array(graphs)
    labels = np.array(labels).squeeze()  # shape N x 12
  #  if target_param is not False:  # regression over a specific target, not all 12 elements
  #      labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1


    for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][14])
           # graph3d, graph3d2, _, _ = high_order3(G, target)
            graph3d,  _, _ = high_order2(G, target)
            adj_power = A_power(graphs[i][14])
            graphs3d.append(graph3d)
           # graphs3d2.append(graph3d2)
            adj_powers.append(adj_power)
           # graphs[i][13] = normalize(graphs[i][13])
           # graphs[i][14] = normalize(graphs[i][14])

    graphs3d = np.array(graphs3d)
    #graphs3d2 = np.array(graphs3d2)
    adj_powers = np.array(adj_powers)
    for i in range(graphs3d.shape[0]):
            graphs3d[i] = np.expand_dims(graphs3d[i], axis=0)
            adj_powers[i] = np.expand_dims(adj_powers[i], axis=0)
           # graphs3d2[i] = np.expand_dims(graphs3d2[i], axis=0)
           # graphs3d[i] = np.concatenate((graphs3d[i], graphs3d2[i], adj_powers[i]), axis=0)
            graphs3d[i] = np.concatenate((graphs3d[i], adj_powers[i]), axis=0)

    return  graphs, labels, graphs3d

def load_qm9_gnn3(target_param,target_shape):
    train_graphs2d, train_labels , train_graphs3d= load_qm9_aux_gnn3('train', target_param,target_shape)
    val_graphs2d, val_labels, val_graphs3d = load_qm9_aux_gnn3('val', target_param,target_shape)
    test_graphs2d, test_labels, test_graphs3d = load_qm9_aux_gnn3('test', target_param,target_shape)
    return train_graphs2d, train_labels, train_graphs3d,  val_graphs2d,val_labels, val_graphs3d,  test_graphs2d, test_labels, test_graphs3d



def gnn1(ds_name, target_shape):
    graphs = []
    labels = []
    if ds_name == 'syn':
        target = motif(target_shape)

        #  graph_dict=dict(zip([5,6,6, 6, 7,8, 9, 9, 10,10], [0.7,0.4,0.5, 0.6, 0.4,0.4,0.4, 0.3, 0.4, 0.3]))
        graph_dict = dict(zip([8, 9, 9, 10, 10, 11, 11, 12, 13], [0.3, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.2, 0.2]))
        num_rep = [50, 50, 50, 50, 100, 100, 100, 100, 100, 100]
        for num, (k, v) in zip(num_rep, graph_dict.items()):
            for s in range(num):
                G = nx.erdos_renyi_graph(k, v, seed=s, directed=False)
                if nx.is_connected(G):
                    graph, label, _ = high_order(G, target)
                    graphs.append(graph)
                    labels.append(label)
        graphs = np.array(graphs)
        labels = np.array(labels)
        graphs = tf.ragged.constant(graphs).to_tensor().eval(session=tf.Session())
    else:
        target = motif(target_shape)
        directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
        with open(directory, "r") as data:
            num_graphs = int(data.readline().rstrip().split(" ")[0])
            for i in range(num_graphs):
                graph_meta = data.readline().rstrip().split(" ")
                num_vertex = int(graph_meta[0])
                curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
                labels.append(int(graph_meta[1]))  # ori
                for j in range(num_vertex):
                    vertex = data.readline().rstrip().split(" ")
                    if NUM_LABELS[ds_name] != 0:
                        curr_graph[j, j, int(vertex[0]) + 1] = 1.
                    for k in range(2, len(vertex)):
                        curr_graph[j, int(vertex[k]), 0] = 1.
                #      curr_graph = noramlize_graph(curr_graph)
                graphs.append(curr_graph)
        graphs = np.array(graphs)

        for i in range(graphs.shape[0]):
            graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # use only A
            G = nx.from_numpy_array(graphs[i][0])
            graph, _, _ = high_order(G, target)
            graphs[i] = np.expand_dims(graph, axis=0)

    return graphs, np.array(labels)












if __name__ == '__main__':
    graphs, labels = load_dataset("MUTAG")
    a, b = get_train_val_indexes(1, "MUTAG")
    print(np.transpose(graphs[a[0]], [1, 2, 0])[0])
