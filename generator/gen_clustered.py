import time as t
import numpy as np
import random as rand
import libs.utilities as utl
from libs.concepts import Concept


def gen_clustered(concept, dim=27, channels_full=17):

    samples, _, m = concept.dataset_shifted()
    encoder_model = concept.model_matrix_encoder()
    sample_threshold = concept.sample_threshold

    # total is twice the size, half for positive pairs, half for negative ones
    pairs_per_sample = 5

    # takes a minute or two...
    print('start encoding lists...')
    encodings_list = []

    predictions = encoder_model.predict(samples)
    assert predictions.shape == (m, dim, dim, channels_full)

    for i in range(m):
        if i % 100 == 0:
            print('\tencoding is at', i)

        sample = samples[i, :, :, 0]
        prediction = predictions[i]
        encoding_list = utl.get_embeddings_from_prediction(prediction, sample, dim=dim, threshold=sample_threshold)
        encodings_list.append(encoding_list)

    print('start pairing...')
    other_indexes = rand.sample(range(m), m)
    assert len(other_indexes) == m

    # m samples, pps positive and pps negative pairs, two encodings in a pair, 17-dim vector encoding
    pairs = np.zeros((m, 2 * pairs_per_sample, 2, channels_full))

    for i in range(m):
        if i % 100 == 0:
            print('\tpairing is at', i)

        encodings = encodings_list[i]
        other_encodings = encodings_list[other_indexes[i]]

        if len(encodings) == 0 or len(other_encodings) == 0:
            print('problem with sample i:', i,
                  'encodings shape:', encodings.shape,
                  'other encodings shape:', other_encodings.shape)
            continue

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
    rand.seed(1)

    gen_clustered(Concept.LINE)
    gen_clustered(Concept.ELLIPSE)
    gen_clustered(Concept.BEZIER)
    gen_clustered(Concept.STAR)

    print('end')
