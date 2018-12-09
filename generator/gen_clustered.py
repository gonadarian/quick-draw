import time as t
import numpy as np
import random as rand
import libs.utilities as utl
from libs.concepts import Concept


dim = 27
channels_full = 17

rand.seed(1)


def main(concept):

    samples, _, m = concept.dataset_shifted()
    encoder_model = concept.model_matrix_encoder()

    # total is twice the size, half for positive pairs, half for negative ones
    pairs_per_sample = 5

    # takes a minute or two...
    print('start encoding lists...')
    encoding_lists = []

    for i in range(m):
        if i % 100 == 0:
            print('\tencoding is at', i)

        sample = samples[i, :, :, 0]
        encoding_list = utl.get_embeddings(encoder_model, sample, dim=dim, show=False)
        encoding_lists.append(encoding_list)

    print('start pairing...')
    other_indexes = rand.sample(range(m), m)
    assert len(other_indexes) == m

    # m samples, pps positive and pps negative pairs, two encodings in a pair, 17-dim vector encoding
    pairs = np.zeros((m, 2 * pairs_per_sample, 2, channels_full))

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

    timestamp = int(t.time())
    filename = 'data/{}/{}-clustered-{}-{}x{}x2x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, m, pairs_per_sample * 2, channels_full)
    np.save(filename, pairs)

    print('saved data', pairs.shape)


if __name__ == '__main__':
    main(Concept.LINE)
    main(Concept.ELLIPSE)
    print('end')
