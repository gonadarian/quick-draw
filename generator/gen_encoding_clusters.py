import numpy as np
import random as rand
from keras.models import load_model
import matplotlib.pyplot as plt


def get_encodings(sample, show=False):
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


rand.seed(1)

samples = np.load('data\line_samples_v2_7234x28x28x1.npy')
assert samples.shape == (7234, 28, 28, 1)
m = samples.shape[0]

encoder = load_model('..\models\lines_encoded\lines_mixed_encoded_v2-091-0.000072.hdf5')

pairs_per_sample = 5  # total is twice the size, half for positive pairs, half for negative ones

print('start encoding lists...')  # takes a minute or two...
encoding_lists = []

for i in range(m):
    if i % 100 == 0:
        print('\tencoding is at', i)

    sample = samples[i, :, :, 0]
    encoding_list = get_encodings(sample, False)
    encoding_lists.append(encoding_list)

print('start pairing...')
other_indexes = rand.sample(range(m), m)
assert len(other_indexes) == m

# m samples, pps positive and pps negative pairs, two encodings in a pair, 17-dim vector encoding
pairs = np.zeros((m, 2 * pairs_per_sample, 2, 17))

for i in range(m):
    if i % 100 == 0:
        print('\tpairing is at', i)

    encodings = encoding_lists[i]
    other_encodings = encoding_lists[other_indexes[i]]

    # positive pairs that should match
    pair_indexes = np.random.randint(len(encodings), size=(2, pairs_per_sample))
    pairs[i, :pairs_per_sample, 0, :] = encodings[pair_indexes[0]]
    pairs[i, :pairs_per_sample, 1, :] = encodings[pair_indexes[1]]

    # negative pairs that should not match
    pair_indexes = np.random.randint(len(encodings), size=pairs_per_sample)
    other_pair_indexes = np.random.randint(len(other_encodings), size=pairs_per_sample)
    pairs[i, pairs_per_sample:, 0, :] = encodings[pair_indexes]
    pairs[i, pairs_per_sample:, 1, :] = other_encodings[other_pair_indexes]


np.save('data\encoding_clusters_v2_{}x10x2x17.npy'.format(m), pairs)
print('saved data', m)

print('end')
