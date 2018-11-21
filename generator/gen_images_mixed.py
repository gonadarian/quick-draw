import numpy as np
import random as rand


rand.seed(1)


def mix_samples(samples, encodings, index_1, index_2):
    print('mixing', index_1, 'and', index_2)

    sample_1 = samples[index_1, ...]
    sample_2 = samples[index_2, ...]
    sample_mix = sample_1 + sample_2
    normalizer = np.array(sample_mix)
    normalizer[normalizer == 0] = 1
    sample_mix = sample_mix / normalizer

    encoding_1 = encodings[index_1, ...]
    encoding_2 = encodings[index_2, ...]
    encoding_mix = encoding_1 + encoding_2
    encoding_mix = encoding_mix / normalizer

    return sample_mix, encoding_mix


def main():

    samples = np.load('data\line_samples_v2_7234x28x28x1.npy')  # TODO use datasets load method
    encodings = np.load('data\line_encodings_v2_7234x28x28x16.npy')  # TODO use datasets load method
    assert samples.shape == (7234, 28, 28, 1)
    assert encodings.shape == (7234, 28, 28, 17)
    assert samples.shape[0] == encodings.shape[0]

    m = samples.shape[0]

    indexes = rand.sample(range(m), m)
    assert len(indexes) == m

    samples_mix = np.zeros((m, 28, 28, 1))
    encodings_mix = np.zeros((m, 28, 28, 17))

    for index in range(m):
        sample, encoding = mix_samples(samples, encodings, index, indexes[index])
        assert sample.shape == (28, 28, 1)
        assert encoding.shape == (28, 28, 17)
        samples_mix[index, ...] = sample
        encodings_mix[index, ...] = encoding

    np.save('data\line_mixed_samples_v2_{}x28x28x1.npy'.format(m), samples_mix)
    np.save('data\line_mixed_encodings_v2_{}x28x28x17.npy'.format(m), encodings_mix)


if __name__ == '__main__':
    main()
    print('end')
