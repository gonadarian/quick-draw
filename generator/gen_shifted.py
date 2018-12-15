import time as t
import numpy as np
import random as rand
import libs.generators as gens
from libs.concepts import Concept


def gen_shifted(concept, density, dim=27, channels=14, channels_full=17):

    shift_matrix = gens.get_shift_matrix(dim)
    assert shift_matrix.shape == (1, dim, dim, 2)

    x, m = concept.dataset_centered()
    print("x: ", x.shape)

    _, encoder_model, _ = concept.model_autoencoder()
    encoding_list = encoder_model.predict(x)
    assert encoding_list.shape == (m, 1, 1, channels)

    count = 0
    x_list = np.zeros((1000, dim, dim, 1))
    y_list = np.zeros((1000, dim, dim, channels_full))

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

            if len(x_list) == count:
                x_list.resize((count + 1000, dim, dim, 1), refcheck=False)
                y_list.resize((count + 1000, dim, dim, channels_full), refcheck=False)

            y_list[count] = y.reshape((1, dim, dim, channels_full))
            x_list[count] = shifted_sample.reshape((1, dim, dim, 1))

            count += 1

        print("done", i, "with", len(shifted_sample_list), "samples")

    print("total count of samples generated is:", count)

    x = np.array(x_list, copy=False)
    print('sample shape:', x.shape)
    x = x[:count]
    print('sample shape after crop:', x.shape)
    assert x.shape[1:] == (dim, dim, 1)

    y = np.array(y_list, copy=False)
    print('encoding shape:', y.shape)
    y = y[:count]
    print('encoding shape after crop:', y.shape)
    assert y.shape[1:] == (dim, dim, channels_full)

    timestamp = int(t.time())

    filename = 'data/{}/{}-shifted-samples-{}-{}x{}x{}x1.npy'
    filename = filename.format(concept.code, concept.code, timestamp, x.shape[0], dim, dim)
    np.save(filename, x)

    filename = 'data/{}/{}-shifted-encodings-{}-{}x{}x{}x{}.npy'
    filename = filename.format(concept.code, concept.code, timestamp, x.shape[0], dim, dim, channels_full)
    np.save(filename, y)


if __name__ == '__main__':
    rand.seed(1)

    gen_shifted(Concept.LINE, density=0.1)
    gen_shifted(Concept.ELLIPSE, density=0.02)
    gen_shifted(Concept.BEZIER, density=0.01)

    print('done all')
