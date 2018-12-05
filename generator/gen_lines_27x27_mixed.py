import numpy as np
import random as rand
import libs.generators as gens


dim = 27
channels_full = 17

rand.seed(1)


def main():

    # TODO use datasets load method
    samples = np.load('data\lines\line_27x27_samples_v1_5815x27x27x1.npy')
    encodings = np.load('data\lines\line_27x27_encodings_v1_5815x27x27x17.npy')
    assert samples.shape == (5815, dim, dim, 1)
    assert encodings.shape == (5815, dim, dim, channels_full)
    assert samples.shape[0] == encodings.shape[0]

    m = samples.shape[0]

    indexes = rand.sample(range(m), m)
    assert len(indexes) == m

    samples_mix = np.zeros((m, dim, dim, 1))
    encodings_mix = np.zeros((m, dim, dim, channels_full))

    for index in range(m):
        sample, encoding = gens.mix_samples(samples, encodings, index, indexes[index])
        assert sample.shape == (dim, dim, 1)
        assert encoding.shape == (dim, dim, channels_full)
        samples_mix[index, ...] = sample
        encodings_mix[index, ...] = encoding

    filename = 'data\lines\line_27x27_mixed_samples_v1_{}x{}x{}x1.npy'.format(m, dim, dim)
    np.save(filename, samples_mix)

    filename = 'data\lines\line_27x27_mixed_encodings_v1_{}x{}x{}x{}.npy'.format(m, dim, dim, channels_full)
    np.save(filename, encodings_mix)


if __name__ == '__main__':
    main()
    print('end')
