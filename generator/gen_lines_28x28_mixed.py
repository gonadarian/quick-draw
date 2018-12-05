import numpy as np
import random as rand
import libs.generators as gens


rand.seed(1)


def main():

    # TODO use datasets load method
    samples = np.load('data\line_samples_v2_7234x28x28x1.npy')
    encodings = np.load('data\line_encodings_v2_7234x28x28x16.npy')
    assert samples.shape == (7234, 28, 28, 1)
    assert encodings.shape == (7234, 28, 28, 17)
    assert samples.shape[0] == encodings.shape[0]

    m = samples.shape[0]

    indexes = rand.sample(range(m), m)
    assert len(indexes) == m

    samples_mix = np.zeros((m, 28, 28, 1))
    encodings_mix = np.zeros((m, 28, 28, 17))

    for index in range(m):
        sample, encoding = gens.mix_samples(samples, encodings, index, indexes[index])
        assert sample.shape == (28, 28, 1)
        assert encoding.shape == (28, 28, 17)
        samples_mix[index, ...] = sample
        encodings_mix[index, ...] = encoding

    np.save('data\line_mixed_samples_v2_{}x28x28x1.npy'.format(m), samples_mix)
    np.save('data\line_mixed_encodings_v2_{}x28x28x17.npy'.format(m), encodings_mix)


if __name__ == '__main__':
    main()
    print('end')
