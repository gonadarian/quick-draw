import math as m
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def get_embeddings(encoder, sample, threshold=1, show=False):
    # prepare sample
    assert sample.shape == (28, 28)
    sample = sample.reshape((1, 28, 28, 1))

    # get prediction
    y_pred = encoder.predict(sample)
    assert y_pred.shape == (1, 28, 28, 17)
    y_pred = y_pred[0]

    # extract relevant pixels
    sample = sample.reshape(28, 28)
    idx_sample = np.argwhere(sample >= threshold)
    assert idx_sample.shape[1:] == (2, )
    embeddings = y_pred[idx_sample[:, 0], idx_sample[:, 1], :]

    # convert from relative positions which can't be compared,
    # to absolute ones suitable for k-means clustering
    centers = idx_sample[:, [1, 0]] / 27 - .5 + embeddings[:, 1:3]
    embeddings[:, 1:3] = centers

    if show:
        plt.imshow(sample)
        plt.show()

    return embeddings


def calculate_cluster_matrix(model, embeddings):
    n = embeddings.shape[0]
    assert embeddings.shape == (n, 17)

    x = np.zeros((n, n, 34))
    x[:, :, :17] = embeddings.reshape((n, 1, 17))
    x[:, :, 17:] = embeddings.reshape((1, n, 17))
    x = x.reshape((n ** 2, 34))

    y = model.predict(x)
    assert y.shape == (n ** 2, 1)
    y = y.reshape((n, n))

    # data improvement, matrix should be symmetric
    y += y.T
    y /= 2
    y -= 0.4  # make it more strict, but this is ugly, maybe ln(Y) + 1?

    return y


def extract_clusters(cluster_matrix, debug=False):
    n = cluster_matrix.shape[0]
    assert cluster_matrix.shape == (n, n)
    cluster_matrix = np.rint(cluster_matrix).astype(int)

    indexes = []
    for row in range(n):
        row_indexes = np.argwhere(cluster_matrix[row, :] == 1)[:, 0]
        indexes.append(row_indexes)
        if debug:
            print('row', row, '\t:', '\t'.join(list(map(lambda x: str(x), row_indexes))))

    clusters = []
    for row, row_indexes in enumerate(indexes):
        row_index_set = set(row_indexes)
        found = False
        for cluster in clusters:
            if len(row_index_set.intersection(cluster)) != 0:
                cluster |= row_index_set
                found = True
                break
        if not found:
            clusters.append(row_index_set)

    return clusters


def show_elbow_curve(encodings, show=False):
    count = encodings.shape[0]
    distance_list = list()
    n_cluster = range(1, min(count - 1, 20))

    for n in n_cluster:
        kmeans = KMeans(n_clusters=n, random_state=0).fit(encodings)
        print('labels:', kmeans.labels_)
        distance = np.average(np.min(cdist(encodings, kmeans.cluster_centers_, 'euclidean'), axis=1))
        print('calculated distance:', distance)
        distance_list.append(distance)

    if show:
        plt.plot(n_cluster, distance_list)
        plt.title('elbow curve')
        plt.show()


def gen_image(decoder, encoding, center, show=False):
    assert encoding.shape == (14, )

    # this is a 14-number encoding for one of the lines in the test set
    encoding = np.reshape(encoding, (1, 1, 1, 14))
    image = decoder.predict(encoding)
    assert image.shape == (1, 28, 28, 1)
    image = image.reshape(28, 28)

    from_row = 28 + center[1]
    from_col = 28 + center[0]
    shifted = np.zeros((84, 84))
    shifted[from_row:from_row+28, from_col:from_col+28] = image
    shifted = shifted[28:56, 28:56]
    assert shifted.shape == (28, 28)

    if show:
        plt.gray()
        plt.imshow(shifted)
        plt.show()

    return shifted


def extract_encoding_and_center(cluster):
    center = cluster[1:3]
    center = np.rint(center * 27).astype(int)
    encoding = cluster[3:17]

    return encoding, center


def decode_clustered_embeddings(decoder, embeddings, n_clusters=1, show=False):
    assert embeddings.shape[1:] == (17, )

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    images = []

    for i in range(n_clusters):
        cluster = kmeans.cluster_centers_[i]
        encoding, center = extract_encoding_and_center(cluster)
        print('cluster center:', center)
        print('cluster encoding:', encoding)
        image = gen_image(decoder, encoding, center, show)
        images.append(image)

    return images


def show_clusters(input_image, cluster_images):
    assert input_image.shape == (28, 28)
    n = len(cluster_images)
    cluster_mix = np.array(cluster_images)
    assert cluster_mix.shape == (n, 28, 28)
    cluster_mix = np.amax(cluster_mix, axis=0)
    assert cluster_mix.shape == (28, 28)

    cols = n + 2
    fig = plt.figure(figsize=(1, cols))

    # display regenerated lines
    for i in range(n):
        fig.add_subplot(1, cols, i + 1)
        plt.imshow(cluster_images[i])

    # display combination of all cluster images
    fig.add_subplot(1, cols, n + 1)
    plt.imshow(cluster_mix)

    # display original source image
    fig.add_subplot(1, cols, n + 2)
    plt.imshow(input_image)

    plt.show()


def get_adjacency_matrix(images, show=False):
    dim = len(images)
    matrix = np.array(images).reshape((dim, 28 * 28))
    adjacency_matrix = np.dot(matrix, matrix.T)
    adjacency_matrix[range(dim), range(dim)] = 0
    adjacency_matrix = np.log(adjacency_matrix)

    if show:
        print(adjacency_matrix)

    return adjacency_matrix


def get_adjacency_matrix_from_edges(dim, edges, self_connected=True):
    adjacency_matrix = np.zeros((dim, dim))
    adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
    adjacency_matrix += adjacency_matrix.T

    if self_connected:
        diag_indexes = range(dim)
        adjacency_matrix[diag_indexes, diag_indexes] = 1

    return adjacency_matrix


def get_graph_edges(adjacency_matrix):
    # adjacency matrix has duplicate info, so remove one triangle of values
    adjacency_matrix[np.triu_indices(len(adjacency_matrix))] = False

    # get indexes of True values
    rows, cols = np.where(adjacency_matrix)
    edges = list(zip(rows, cols))

    return edges


def draw_graph(edges):
    # extract nodes from graph
    nodes = set([n1 for n1, n2 in edges] + [n2 for n1, n2 in edges])

    # create networkx graph
    g = nx.Graph()

    # add nodes
    for node in nodes:
        g.add_node(node)

    # add edges
    for edge in edges:
        g.add_edge(edge[0], edge[1])

    # draw graph
    pos = nx.shell_layout(g)
    nx.draw(g, pos)

    # show graph
    plt.show()


def get_regions(region_count=8, show=False):
    region_range = 2 * m.pi / region_count

    region_borders = np.zeros((region_count+2,))
    region_borders[0] = -m.pi
    for i in range(region_count):
        border = -m.pi + region_range / 2 + i * region_range
        region_borders[i + 1] = border
    region_borders[region_count + 1] = +m.pi

    regions = np.ones((region_count+1, 3))
    regions[:, 0] = region_borders[:-1]
    regions[:, 1] = region_borders[1:]
    regions[:-1, 2] = range(1, region_count + 1)

    if show:
        print('borders:', region_borders)
        print('regions:', regions)

    return regions


def get_region_matrix(nodes, regions, show=False, debug=False):
    node_count = len(nodes)
    region_matrix = np.zeros((node_count, node_count, 5)) if debug else np.zeros((node_count, node_count))

    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                # center node uses reserved region 0
                if debug:
                    region_matrix[i, j, :] = [0, 0, m.nan, m.nan, 0]
                else:
                    region_matrix[i, j] = 0
                continue

            dx = nodes[j, 1] - nodes[i, 1]
            dy = nodes[j, 2] - nodes[i, 2]
            rad = m.atan2(dy, dx)
            deg = rad * 180 / m.pi
            region_index = np.logical_and(regions[:, 0] <= rad, regions[:, 1] >= rad)
            region_index = np.argwhere(region_index).reshape((1, ))
            region = regions[region_index].reshape((3, ))

            if show:
                print('region:', i, j, region)

            if debug:
                region_matrix[i, j, 0] = dx
                region_matrix[i, j, 1] = dy
                region_matrix[i, j, 2] = rad
                region_matrix[i, j, 3] = deg
                region_matrix[i, j, 4] = region[2]
            else:
                region_matrix[i, j] = region[2]

    return region_matrix[:, :, 4] if debug else region_matrix


def get_matrix_transformation(adjacency_matrix, region_matrix):
    dim = len(adjacency_matrix)
    assert adjacency_matrix.shape == (dim, dim)
    assert region_matrix.shape == (dim, dim)

    edge_indexes = np.argwhere(adjacency_matrix == 1)
    node_indexes = edge_indexes[:, 1]
    row_indexes = edge_indexes[:, 0]
    column_indexes = region_matrix[edge_indexes[:, 0], edge_indexes[:, 1]].astype(dtype=np.uint8)

    edge_count = np.count_nonzero(adjacency_matrix) // 2
    assert len(row_indexes) == edge_count * 2
    assert len(column_indexes) == edge_count * 2
    assert len(node_indexes) == edge_count * 2

    return row_indexes, column_indexes, node_indexes


def get_vector_transformation(adjacency_matrix, region_matrix):
    row_indexes, column_indexes, node_indexes = get_matrix_transformation(adjacency_matrix, region_matrix)
    vector_indexes = row_indexes * 9 + column_indexes

    return vector_indexes, node_indexes
