import numpy as np
import random as rand
import libs.models as mdls
import libs.datasets as ds
import libs.generators as gens


dim = 27
channels = 14
channels_full = 17

rand.seed(1)


def main():
    shift_matrix = gens.get_shift_matrix(dim)
    assert shift_matrix.shape == (1, dim, dim, 2)

    x = ds.load_images_line_27x27_centered()
    m = x.shape[0]
    print("x: ", x.shape)

    y = np.zeros((m, dim, dim, channels_full))
    print("y: ", y.shape)

    autoencoder_model = mdls.load_autoencoder_model_27x27()
    autoencoder_model.outputs = [autoencoder_model.layers[8].output]

    x_list = []
    y_list = []

    for i in range(m):
        x_sample = [x[[i], ...]]
        encoding = autoencoder_model.predict(x_sample)
        assert encoding.shape == (1, 1, 1, channels)
        encoding = encoding.reshape((1, 1, channels))

        x_sample = x_sample[0][0, :, :, 0]
        assert x_sample.shape == (dim, dim)

        samples = gens.generated_shifted_samples(x_sample, dim, density=0.1)
        for sample, shift_row, shift_col in samples:
            assert sample.shape == (dim, dim)
            sample = sample.reshape((dim, dim, 1))

            shift = np.zeros((1, 1, 1, 2))
            shift[0, 0, 0, :] = [shift_col / (dim - 1), shift_row / (dim - 1)]  # 'dim' points, but 'dim-1' ranges
            shift = shift_matrix + shift

            y = np.zeros((dim, dim, channels_full))
            y[:, :, [0]] = sample
            y[:, :, 1:3] = sample * shift
            y[:, :, 3:channels_full] = sample * encoding

            y_list.append(y.reshape((1, dim, dim, channels_full)))
            x_list.append(sample.reshape((1, dim, dim, 1)))

        print("done", i, "with", len(samples), "samples")

    x = np.concatenate(x_list)
    print('sample shape:', x.shape)
    assert x.shape[1:] == (dim, dim, 1)

    y = np.concatenate(y_list)
    print('encoding shape:', y.shape)
    assert y.shape[1:] == (dim, dim, channels_full)

    np.save('data\lines\line_27x27_samples_v1_{}x{}x{}x1.npy'.format(x.shape[0], dim, dim), x)
    np.save('data\lines\line_27x27_encodings_v1_{}x{}x{}x{}.npy'.format(y.shape[0], dim, dim, channels_full), y)


if __name__ == '__main__':
    main()
    print('done all')
