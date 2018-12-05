import numpy as np


def get_shift_matrix(dim):
    matrix = np.zeros((dim, dim, 2))
    row = np.arange(0, dim).reshape((1, dim, 1))
    col = np.arange(0, dim).reshape((dim, 1, 1))

    matrix[:, :, [0]] = row
    matrix[:, :, [1]] = col
    matrix -= (dim - 1) / 2
    matrix /= -(dim - 1)
    matrix = matrix.reshape((1, dim, dim, 2))

    return matrix


def mix_samples(samples, encodings, index_1, index_2):
    print('mixing samples', index_1, 'and', index_2)

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
