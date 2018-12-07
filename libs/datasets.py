import os
import numpy as np


def load(filename):
    path = os.path.join(os.path.dirname(__file__), '../generator/data/', filename)
    dataset = np.load(path)
    return dataset


def load_images_line_27x27_centered():
    x = load('line/line_centered_1544186065_364x27x27.npy')
    m = x.shape[0]
    assert m == 364
    assert x.shape == (m, 27, 27)

    x = x.astype('float32') / 255.
    x = np.reshape(x, (len(x), 27, 27, 1))
    assert x.shape == (m, 27, 27, 1)

    return x, m


def load_images_line_27x27_shifted():
    filename = 'line/line_shifted_samples_1544171724_5815x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 5815
    assert x.shape == (m, 27, 27, 1)

    filename = 'line/line_shifted_encodings_1544171724_5815x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_line_27x27_mixed():
    filename = 'line/line_mixed_samples_1544185269_5815x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 5815
    assert x.shape == (m, 27, 27, 1)

    filename = 'line/line_mixed_encodings_1544185269_5815x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_line_27x27_clustered():
    filename = 'line/line_clusters_1544187266_5815x10x2x17.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 5815
    assert x.shape == (m, 10, 2, 17)

    # TODO pair count, 10, should be a parameter of this method
    x = x.reshape((m * 10, 34))
    m = x.shape[0]

    # TODO this code is used in more places, should be extracted
    y = np.tile(np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]), m // 10)
    y = y.reshape((m, 1))
    assert y.shape == (m, 1)

    return x, y, m


def load_images_ellipse_27x27_centered():
    x = load('ellipse/ellipse_centered_1544185820_2028x27x27.npy')
    m = x.shape[0]
    assert m == 2028
    assert x.shape == (m, 27, 27)

    x = x.astype('float32') / 255.
    x = np.reshape(x, (m, 27, 27, 1))
    assert x.shape == (m, 27, 27, 1)

    return x, m


def load_images_ellipse_27x27_shifted():
    filename = 'ellipse/ellipse_shifted_samples_1544172002_8592x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 8592
    assert x.shape == (m, 27, 27, 1)

    filename = 'ellipse/ellipse_shifted_encodings_1544172002_8592x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_ellipse_27x27_mixed():
    filename = 'ellipse/ellipse_mixed_samples_1544185490_8592x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 8592
    assert x.shape == (m, 27, 27, 1)

    filename = 'ellipse/ellipse_mixed_encodings_1544185490_8592x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_ellipse_27x27_clustered():
    filename = 'ellipse/ellipse_clusters_1544193289_8592x10x2x17.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 8592
    assert x.shape == (m, 10, 2, 17)

    # TODO pair count, 10, should be a parameter of this method
    x = x.reshape((m * 10, 34))
    m = x.shape[0]

    # TODO this code is used in more places, should be extracted
    y = np.tile(np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]), m // 10)
    y = y.reshape((m, 1))
    assert y.shape == (m, 1)

    return x, y, m


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
