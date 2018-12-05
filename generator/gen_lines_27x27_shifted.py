import math
import numpy as np
import random as rand
import models as mdls
import datasets as ds
import libs.generators as gens


dim = 27
channels = 14
channels_full = 17

rand.seed(1)


def generated_shifted_samples(sample, density=0.3):
    assert sample.shape == (dim, dim)

    [empty_rows, empty_cols] = np.amin(np.where(sample == 1), axis=1)
    sub_image = sample[empty_rows:dim - empty_rows, empty_cols:dim - empty_cols]

    # lines are not centered after all.... so +1/-1 on couple of places :(
    rows = dim - 2 * empty_rows
    cols = dim - 2 * empty_cols

    max_image_count = 0 if empty_rows == 0 and empty_cols == 0 else\
        2 * empty_rows + 2 * empty_cols if empty_rows == 0 or empty_cols == 0 else\
        4 * empty_rows * empty_cols

    image_count = math.ceil(max_image_count * density)
    samples = [(sample, 0, 0)]

    for i in range(image_count):
        shift_row = 0 if empty_rows == 0 else rand.randint(-empty_rows, empty_rows)
        shift_col = 0 if empty_cols == 0 else rand.randint(-empty_cols, empty_cols)
        from_row = empty_rows + shift_row
        from_col = empty_cols + shift_col
        shifted_sample = np.zeros((dim, dim))
        shifted_sample[from_row:from_row+rows, from_col:from_col+cols] = sub_image
        samples.append((shifted_sample, shift_row, shift_col))

    return samples


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

        samples = generated_shifted_samples(x_sample, density=0.1)
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
