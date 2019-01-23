import math as m
import numpy as np
import random as rand
import networkx as nx
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from skimage.measure import compare_ssim as ssim


# TODO use get_embeddings_from_prediction internally, don't duplicate code, or do something eve better!
def get_embeddings(encoder, sample, dim=27, threshold=1, show=False):
    # prepare sample
    assert sample.shape == (dim, dim)
    sample = sample.reshape((1, dim, dim, 1))

    # get show_predictions
    prediction = encoder.predict(sample)  # TODO predict for all samples at once, not once per sample
    assert prediction.shape == (1, dim, dim, 17)
    prediction = prediction[0]

    # extract relevant pixels
    sample = sample.reshape(dim, dim)
    idx_sample = np.argwhere(sample >= threshold)
    assert idx_sample.shape[1:] == (2, )
    embeddings = prediction[idx_sample[:, 0], idx_sample[:, 1], :]

    # convert from relative positions which can't be compared,
    # to absolute ones suitable for k-means clustering
    centers = idx_sample[:, [1, 0]] / (dim - 1) - .5 + embeddings[:, 1:3]
    embeddings[:, 1:3] = centers

    if show:
        plt.imshow(sample)
        plt.show()

    return embeddings


def get_embeddings_from_prediction(prediction, sample, dim=27, threshold=1, show=False):
    assert sample.shape == (dim, dim)
    assert prediction.shape == (dim, dim, 17)

    # extract relevant pixels
    idx_sample = np.argwhere(sample >= threshold)
    assert idx_sample.shape[1:] == (2, )
    embeddings = prediction[idx_sample[:, 0], idx_sample[:, 1], :]

    # convert from relative positions which can't be compared,
    # to absolute ones suitable for k-means clustering
    centers = idx_sample[:, [1, 0]] / (dim - 1) - .5 + embeddings[:, 1:3]
    embeddings[:, 1:3] = centers

    if show:
        plt.imshow(sample)
        plt.show()

    return embeddings


def calculate_cluster_matrix(clustering_model, embeddings):
    n = embeddings.shape[0]
    assert embeddings.shape == (n, 17)

    # combine each of N embeddings with all other embeddings in a N*N matrix
    x = np.zeros((n, n, 34))
    # use broadcasting to send embeddings as columns and then as rows
    x[:, :, :17] = embeddings.reshape((n, 1, 17))
    x[:, :, 17:] = embeddings.reshape((1, n, 17))
    # each cell is now combination of two embeddings
    x = x.reshape((n ** 2, 34))

    y = clustering_model.predict(x)
    assert y.shape == (n ** 2, 1)
    y = y.reshape((n, n))

    # data improvement, matrix should be symmetric
    y = (y + y.T) / 2

    # makes it more strict, but this is ugly, maybe ln(Y) + 1?
    y -= 0.4

    return y


def extract_clusters_v1(cluster_matrix, debug=False):
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


def extract_clusters(cluster_matrix, debug=False):
    # TODO this is both ugly and brittle
    cluster_match_threshold = 4

    n = cluster_matrix.shape[0]
    assert cluster_matrix.shape == (n, n)
    cluster_matrix = np.rint(cluster_matrix).astype(int)

    indexes = []
    for row in range(n):
        row_indices = np.argwhere(cluster_matrix[row, :] == 1)[:, 0]
        indexes.append(row_indices)
        if debug:
            print('row', row, '\t:', '\t'.join(list(map(lambda x: str(x), row_indices))))

    clusters = []
    for row, row_indices in enumerate(indexes):
        print('indices for row {} are {}'.format(row, row_indices))
        row_index_set = set(row_indices)
        best_match = (-1, 0)

        for cluster_index, cluster in enumerate(clusters):
            match = len(row_index_set.intersection(cluster))
            print('\tfor cluster {} match: {}'.format(cluster_index, match))
            if match > cluster_match_threshold and match > best_match[1]:
                best_match = (cluster_index, match)

        cluster_index = best_match[0]
        if cluster_index >= 0:
            clusters[cluster_index] |= row_index_set
            print('existing cluster extended: {}'.format(clusters[cluster_index]))
        else:
            print('new cluster added: {}'.format(row_index_set))
            clusters.append(row_index_set)

    return clusters


def show_distances(encodings):
    n = encodings.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = spatial.distance.cosine(encodings[i, :], encodings[j, :])

    for idx in range(n):
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), distances[idx, :]))), '\n')


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


def gen_image(decoder, encoding, center, dim=27, show=False):
    assert encoding.shape == (14, )

    # this is a 14-number encoding for one of the lines in the test set
    encoding = np.reshape(encoding, (1, 1, 1, 14))
    image = decoder.predict(encoding)
    assert image.shape == (1, dim, dim, 1)
    image = image.reshape(dim, dim)

    from_row = dim + center[1]
    from_col = dim + center[0]
    shifted = np.zeros((dim * 3, dim * 3), dtype=np.float32)
    shifted[from_row:from_row + dim, from_col:from_col + dim] = image
    shifted = shifted[dim:dim * 2, dim:dim * 2]
    assert shifted.shape == (dim, dim)

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


def decode_clustered_embeddings(decoder, embeddings, n_clusters=1, dim=27, show=False):
    assert embeddings.shape[1:] == (17, )

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    images = []

    for i in range(n_clusters):
        cluster = kmeans.cluster_centers_[i]
        encoding, center = extract_encoding_and_center(cluster)
        print('cluster center:', center)
        print('cluster encoding:', encoding)
        image = gen_image(decoder, encoding, center, dim=dim, show=show)
        images.append(image)

    return images


def calc_similarity(image_target, image_source, algorithm='dot'):
    if algorithm == 'dot':
        similarity = np.dot(image_target.reshape((-1,)), image_source.reshape((-1,)))
        similarity /= np.sum(image_source)

    elif algorithm == 'mse':
        mse = np.sum((image_target - image_source) ** 2)
        # mse /= float(image_target.shape[0] * image_target.shape[1])
        mse /= np.sum(image_source)
        similarity = 1 - mse

    elif algorithm == 'ssim':
        similarity = ssim(image_target, image_source)

    elif algorithm == 'gauss':
        image_target = gaussian_filter(image_target, sigma=1)
        similarity = calc_similarity(image_target, image_source, algorithm='mse')

    else:
        assert False, '[calc_similarity] bad algorithm'

    return similarity


def show_clusters(input_image, cluster_images, cluster_weights=None, dim=27):
    if len(cluster_images) == 0:
        print('[show_clusters] WARNING: no clusters provided')
        return

    if cluster_weights is None:
        cluster_weights = ['?'] * len(cluster_images)

    print('[show_clusters] input_image.shape', input_image.shape)
    assert input_image.shape == (dim, dim)
    n = len(cluster_images)
    print('[show_clusters] len(cluster_images)', n)
    cluster_mix = np.array(cluster_images)
    print('[show_clusters] cluster_images.shape', cluster_mix.shape)
    assert cluster_mix.shape == (n, dim, dim)
    cluster_mix = np.amax(cluster_mix, axis=0)
    assert cluster_mix.shape == (dim, dim)

    cols = n + 2
    fig = plt.figure(figsize=(1, cols))

    # display regenerated concepts
    for i, (cluster_image, cluster_weight) in enumerate(zip(cluster_images, cluster_weights)):
        similarity = calc_similarity(input_image, cluster_image, algorithm='gauss')
        print('[show_clusters] cluster {} has similarity {}'.format(i, similarity))
        axis = fig.add_subplot(1, cols, i + 1)
        axis.set_title('s:{:.2f} w:{}'.format(similarity, cluster_weight))
        plt.imshow(cluster_image)

    # display combination of all cluster images
    fig.add_subplot(1, cols, n + 1)
    plt.imshow(cluster_mix)

    # display original source image
    fig.add_subplot(1, cols, n + 2)
    plt.imshow(input_image)

    plt.show()


def get_adjacency_matrix(images, dim=27, show=False):
    vertices = len(images)
    matrix = np.array(images).reshape((vertices, dim * dim))
    adjacency_matrix = np.dot(matrix, matrix.T)
    adjacency_matrix[range(vertices), range(vertices)] = 0
    adjacency_matrix = np.log(adjacency_matrix)

    if show:
        print(adjacency_matrix)

    return adjacency_matrix


def get_adjacency_matrix_from_edges(vertices, edge_list, self_connected=True):
    adjacency_matrix = np.zeros((vertices, vertices))

    if len(edge_list) > 0:
        adjacency_matrix[edge_list[:, 0], edge_list[:, 1]] = 1
        adjacency_matrix += adjacency_matrix.T

        if self_connected:
            diagonal_indexes = range(vertices)
            adjacency_matrix[diagonal_indexes, diagonal_indexes] = 1

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


def get_regions(region_count=9, show=False):
    region_range = 2 * m.pi / (region_count - 1)

    region_borders = np.zeros((region_count + 1,))
    region_borders[0] = -m.pi
    for i in range(region_count - 1):
        border = -m.pi + region_range / 2 + i * region_range
        region_borders[i + 1] = border
    region_borders[region_count] = +m.pi

    regions = np.ones((region_count, 3))
    regions[:, 0] = region_borders[:-1]
    regions[:, 1] = region_borders[1:]
    regions[:-1, 2] = range(1, region_count)

    if show:
        print('borders:', region_borders)
        print('regions:', regions)

    return regions


# TODO split this to two functions, one for debug, one regular, or something similar
def get_region_matrix(node_list, region_list, show=False, debug=False):
    vertices = len(node_list)
    region_matrix = np.zeros((vertices, vertices, 5)) if debug else np.zeros((vertices, vertices))

    for i in range(vertices):
        for j in range(vertices):
            if i == j:
                # center node uses reserved region 0
                if debug:
                    region_matrix[i, j, :] = [0, 0, m.nan, m.nan, 0]
                else:
                    region_matrix[i, j] = 0
                continue

            dx = node_list[j, 1] - node_list[i, 1]
            dy = node_list[j, 2] - node_list[i, 2]
            rad = m.atan2(dy, dx)
            deg = rad * 180 / m.pi
            region_index = np.logical_and(region_list[:, 0] <= rad, region_list[:, 1] >= rad)
            region_index = np.argwhere(region_index).reshape((1, ))
            region = region_list[region_index].reshape((3,))

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

    edge_count = np.count_nonzero(adjacency_matrix)
    assert len(row_indexes) == edge_count
    assert len(column_indexes) == edge_count
    assert len(node_indexes) == edge_count

    return row_indexes, column_indexes, node_indexes


def get_vector_transformation(adjacency_matrix, region_matrix):
    row_indexes, column_indexes, node_indexes = get_matrix_transformation(adjacency_matrix, region_matrix)
    vector_indexes = row_indexes * 9 + column_indexes

    return vector_indexes, node_indexes


def show_predictions(model, x, y, n=10, dim=27):
    m = x.shape[0]
    indices = rand.sample(range(1, m), n)

    x = x[indices]
    y = y[indices]
    y_pred = model.predict(x)
    plt.figure(figsize=(n * 3, 4))

    for index in range(n):
        # display original
        ax = plt.subplot(3, n, index + 1)
        original = x[index].reshape(dim, dim)
        plt.imshow(original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(index)

        # display predictions
        ax = plt.subplot(3, n, index + 1 + n)
        predicted = y_pred[index].reshape(dim, dim)
        plt.imshow(predicted)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display target
        ax = plt.subplot(3, n, index + 1 + 2 * n)
        target = y[index].reshape(dim, dim)
        plt.imshow(target - predicted)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
