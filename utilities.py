import numpy as np
import matplotlib.pyplot as plt


def get_embeddings(encoder, sample, show=False):
    # prepare sample
    assert sample.shape == (28, 28)
    sample = sample.reshape((1, 28, 28, 1))

    # get prediction
    y_pred = encoder.predict(sample)
    assert y_pred.shape == (1, 28, 28, 17)
    y_pred = y_pred[0]

    # extract relevant pixels
    sample = sample.reshape(28, 28)
    idx_sample = np.argwhere(sample == 1)
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
    m = embeddings.shape[0]
    assert embeddings.shape == (m, 17)

    X = np.zeros((m, m, 34))
    X[:, :, :17] = embeddings.reshape((m, 1, 17))
    X[:, :, 17:] = embeddings.reshape((1, m, 17))
    X = X.reshape((m ** 2, 34))

    Y = model.predict(X)
    assert Y.shape == (m ** 2, 1)
    Y = Y.reshape((m, m))

    # data improvement, matrix should be symmetric
    Y += Y.T
    Y /= 2
    Y -= 0.4  # make it more strict, but this is ugly, maybe ln(Y) + 1?

    return Y


def extract_clusters(cluster_matrix):
    m = cluster_matrix.shape[0]
    assert cluster_matrix.shape == (m, m)
    cluster_matrix = np.rint(cluster_matrix).astype(int)

    indexes = []
    for row in range(m):
        row_indexes = np.argwhere(cluster_matrix[row, :] == 1)[:, 0]
        indexes.append(row_indexes)
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


# TODO
# def decode_clustered_embeddings():