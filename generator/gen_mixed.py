import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


dim = 27
channels_full = 17

rand.seed(1)


def main(concept):

    samples, encodings, m = concept.dataset_shifted()

    indices = rand.sample(range(m), m)
    assert len(indices) == m

    samples_mix = np.zeros((m, dim, dim, 1))
    encodings_mix = np.zeros((m, dim, dim, channels_full))

    for index in range(m):
        sample, encoding = gens.mix_samples(samples, encodings, index, indices[index])
        assert sample.shape == (dim, dim, 1)
        assert encoding.shape == (dim, dim, channels_full)
        samples_mix[index, ...] = sample
        encodings_mix[index, ...] = encoding

    timestamp = int(t.time())

    filename = 'data/{}/{}-mixed-samples-{}-{}x{}x{}x1.npy'
    filename = filename.format(concept.code, concept.code, timestamp, m, dim, dim)
    np.save(filename, samples_mix)

    filename = 'data/{}/{}-mixed-encodings-{}-{}x{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, m, dim, dim, channels_full)
    np.save(filename, encodings_mix)


if __name__ == '__main__':
    main(Concept.LINE)
    main(Concept.ELLIPSE)
    print('end')
