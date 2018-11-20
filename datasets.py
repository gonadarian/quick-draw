import numpy as np


def load_encoding_clusers():
    x = np.load('generator\data\encoding_clusters_v2_7234x10x2x17.npy')
    assert x.shape == (7234, 10, 2, 17)
    return x


def load_graph_lines():
    nodes = np.load('generator/data/graph_lines.npy')
    assert nodes.shape[1:] == (17, )
    return nodes


def load_graph_edges():
    edges = np.load('generator/data/graph_edges.npy')
    assert edges.shape[1:] == (2, )
    return  edges
