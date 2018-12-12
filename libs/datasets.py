import os
import numpy as np


def load(filename):
    path = os.path.join(os.path.dirname(__file__), '../generator/data/', filename)
    dataset = np.load(path)
    print('LOADED DATASET from path {} with shape'.format(path, dataset.shape))
    return dataset


def load_images_quickdraw(category, dim):
    filename = 'quickdraw/quickdraw-{}.npy'.format(category)
    x = load(filename)
    x = x.reshape((-1, 28, 28))[:, :dim, :dim]
    x = x.astype('float32') / 255.
    assert x.shape[1:] == (dim, dim)

    return x


def load_images_line_27x27_centered():
    x = load('line/line-centered-1544186065-364x27x27.npy')
    m = x.shape[0]
    assert m == 364
    assert x.shape == (m, 27, 27)

    x = x.astype('float32') / 255.
    x = np.reshape(x, (len(x), 27, 27, 1))
    assert x.shape == (m, 27, 27, 1)

    return x, m


def load_images_line_27x27_shifted():
    filename = 'line/line-shifted-samples-1544485663-6242x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 6242
    assert x.shape == (m, 27, 27, 1)

    filename = 'line/line-shifted-encodings-1544485663-6242x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_line_27x27_mixed():
    filename = 'line/line-mixed-samples-1544485778-6242x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 6242
    assert x.shape == (m, 27, 27, 1)

    filename = 'line/line-mixed-encodings-1544485778-6242x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_line_27x27_clustered():
    filename = 'line/line-clustered-1544541841-6242x10x2x17.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 6242
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
    x = load('ellipse/ellipse-centered-1544185820-2028x27x27.npy')
    m = x.shape[0]
    assert m == 2028
    assert x.shape == (m, 27, 27)

    x = x.astype('float32') / 255.
    x = np.reshape(x, (m, 27, 27, 1))
    assert x.shape == (m, 27, 27, 1)

    return x, m


def load_images_ellipse_27x27_shifted():
    filename = 'ellipse/ellipse-shifted-samples-1544172002-8592x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 8592
    assert x.shape == (m, 27, 27, 1)

    filename = 'ellipse/ellipse-shifted-encodings-1544172002-8592x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_ellipse_27x27_mixed():
    filename = 'ellipse/ellipse-mixed-samples-1544185490-8592x27x27x1.npy'
    x = load(filename)
    m = x.shape[0]
    assert m == 8592
    assert x.shape == (m, 27, 27, 1)

    filename = 'ellipse/ellipse-mixed-encodings-1544185490-8592x27x27x17.npy'
    y = load(filename)
    assert y.shape == (m, 27, 27, 17)

    return x, y, m


def load_images_ellipse_27x27_clustered():
    filename = 'ellipse/ellipse-clustered-1544307680-8592x10x2x17.npy'
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


def load_graphs_square_centered():
    vertices_list = load('square/square-centered-vertices-1544545197-114x4x17.npy')
    m = len(vertices_list)
    assert vertices_list.shape == (m, 4, 17)

    mappings_list = load('square/square-centered-mappings-1544545197-114x4x9.npy')
    assert mappings_list.shape == (m, 4, 9)

    return vertices_list, mappings_list, m


def load_graph_lines():
    nodes = load('graph-lines.npy')
    assert nodes.shape[1:] == (17, )
    return nodes


def load_graph_edges():
    edges = load('graph-edges.npy')
    assert edges.shape[1:] == (2, )
    return edges
