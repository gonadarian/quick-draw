import numpy as np
import random as rand
import libs.models as mdls
import libs.utilities as utl

rand.seed(1)


def main():

    samples = np.load('data\line_samples_v2_7234x28x28x1.npy')  # TODO use datasets load method
    assert samples.shape == (7234, 28, 28, 1)
    m = samples.shape[0]

    matrix_encoder_model = mdls.load_matrix_encoder_line_model()

    pairs_per_sample = 5  # total is twice the size, half for positive pairs, half for negative ones

    print('start encoding lists...')  # takes a minute or two...
    encoding_lists = []

    for i in range(m):
        if i % 100 == 0:
            print('\tencoding is at', i)

        sample = samples[i, :, :, 0]
        encoding_list = utl.get_embeddings(matrix_encoder_model, sample, show=False)
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


if __name__ == '__main__':
    main()
    print('end')
