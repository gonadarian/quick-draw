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
    return edges


def load_graph_lines_set():
    nodes = np.load('generator/data/graph_lines_set_v1_146x4x17.npy')
    assert nodes.shape[1:] == (4, 17)
    return nodes


def load_graph_edges_set():
    edges = np.load('generator/data/graph_edges_set_v1_146x4x2.npy')
    assert edges.shape[1:] == (4, 2)
    return edges


def load_graph_mapping_set():
    mappings = np.load('generator/data/graph_mapping_set_v1_146x4x9.npy')
    assert mappings.shape[1:] == (4, 9)
    return mappings


