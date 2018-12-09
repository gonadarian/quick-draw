import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


dim = 27
channels = 14
channels_full = 17

rand.seed(1)


def main(concept, density):

    shift_matrix = gens.get_shift_matrix(dim)
    assert shift_matrix.shape == (1, dim, dim, 2)

    x, m = concept.dataset_centered()
    print("x: ", x.shape)

    autoencoder_model = concept.model_autoencoder()
    autoencoder_model.outputs = [autoencoder_model.layers[8].output]

    encoding_list = autoencoder_model.predict(x)
    assert encoding_list.shape == (m, 1, 1, channels)

    x_list = []
    y_list = []

    for i in range(m):
        encoding = encoding_list[i]
        assert encoding.shape == (1, 1, channels)

        sample = x[i, :, :, 0]
        assert sample.shape == (dim, dim)

        shifted_sample_list = gens.generated_shifted_samples(sample, dim, density=density)
        for shifted_sample, shift_row, shift_col in shifted_sample_list:
            assert shifted_sample.shape == (dim, dim)
            shifted_sample = shifted_sample.reshape((dim, dim, 1))

            shift = np.zeros((1, 1, 1, 2))

            # 'dim' points, but 'dim-1' ranges
            shift[0, 0, 0, :] = [shift_col / (dim - 1), shift_row / (dim - 1)]
            shift = shift_matrix + shift

            y = np.zeros((dim, dim, channels_full))
            y[:, :, [0]] = shifted_sample
            y[:, :, 1:3] = shifted_sample * shift
            y[:, :, 3:channels_full] = shifted_sample * encoding

            y_list.append(y.reshape((1, dim, dim, channels_full)))
            x_list.append(shifted_sample.reshape((1, dim, dim, 1)))

        print("done", i, "with", len(shifted_sample_list), "samples")

    x = np.concatenate(x_list)
    print('sample shape:', x.shape)
    assert x.shape[1:] == (dim, dim, 1)

    y = np.concatenate(y_list)
    print('encoding shape:', y.shape)
    assert y.shape[1:] == (dim, dim, channels_full)

    timestamp = int(t.time())

    filename = 'data/{}/{}-shifted-samples-{}-{}x{}x{}x1.npy'
    filename = filename.format(concept.code, concept.code, timestamp, x.shape[0], dim, dim)
    np.save(filename, x)

    filename = 'data/{}/{}-shifted-encodings-{}-{}x{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, x.shape[0], dim, dim, channels_full)
    np.save(filename, y)


if __name__ == '__main__':
    main(Concept.LINE, density=0.1)
    main(Concept.ELLIPSE, density=0.02)
    print('done all')
