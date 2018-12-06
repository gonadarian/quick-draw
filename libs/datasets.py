import os
import numpy as np


def load(filename):
    path = os.path.join(os.path.dirname(__file__), '../generator/data/', filename)
    dataset = np.load(path)
    return dataset


def load_images_line_centered():
    x = load('line_originals_v2_392x28x28.npy')
    assert x.shape == (392, 28, 28)
    x = x.astype('float32') / 255.
    x = np.reshape(x, (len(x), 28, 28, 1))
    assert x.shape == (392, 28, 28, 1)
    return x


def load_images_line_27x27_centered():
    x = load('lines_27x27/line_27x27_centered_v1_351x27x27.npy')
    assert x.shape == (351, 27, 27)
    x = x.astype('float32') / 255.
    x = np.reshape(x, (len(x), 27, 27, 1))
    assert x.shape == (351, 27, 27, 1)
    return x


def load_encoding_clusters():
    x = load('encoding_clusters_v2_7234x10x2x17.npy')
    assert x.shape == (7234, 10, 2, 17)
    return x


def load_graph_lines():
    nodes = load('graph_lines.npy')
    assert nodes.shape[1:] == (17, )
    return nodes


def load_graph_edges():
    edges = load('graph_edges.npy')
    assert edges.shape[1:] == (2, )
    return edges


def load_graph_lines_set():
    nodes = load('graph_lines_set_v1_146x4x17.npy')
    assert nodes.shape[1:] == (4, 17)
    return nodes


def load_graph_edges_set():
    edges = load('graph_edges_set_v1_146x4x2.npy')
    assert edges.shape[1:] == (4, 2)
    return edges


def load_graph_mapping_set():
    mappings = load('graph_mapping_set_v1_146x4x9.npy')
    assert mappings.shape[1:] == (4, 9)
    return mappings
