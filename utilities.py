import numpy as np
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
    import matplotlib.pyplot as plt

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
