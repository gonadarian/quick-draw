import numpy as np


def get_shift_matrix():
    d = np.zeros((28, 28, 2))
    row = np.arange(0,28).reshape((1, 28, 1))
    col = np.arange(0,28).reshape((28, 1, 1))
    d[:, :, [0]] = row
    d[:, :, [1]] = col
    d -= 13.5
    d /= -27
    d = d.reshape((1, 28, 28, 2))
    return d


d = get_shift_matrix()
assert d.shape == (1, 28, 28, 2)

x = np.load('line_originals_v2_392x28x28.npy')
x = x.astype('float32') / 255.
m = x.shape[0]
x = np.reshape(x, (m, 28, 28, 1))  # adapt this if using `channels_first` image data format
print("x: ", x.shape)

y = np.zeros((m, 28, 28, 16))
print("y: ", y.shape)

from keras.models import load_model
autoencoder = load_model('..\models\lines\model_autoencoder_v2.385-0.0047.hdf5')
autoencoder.outputs = [autoencoder.layers[8].output]


import math
import random

seed = 1
random.seed(seed)


def generated_shifted_samples(sample, density=0.3):
    assert sample.shape == (28, 28)

    [empty_rows, empty_cols] = np.amin(np.where(sample == 1), axis=1)
    sub_image = sample[empty_rows:28 - empty_rows, empty_cols:28 - empty_cols]
    # lines are not centered after all.... so +1/-1 on couple of places :(
    rows = 28 - 2 * empty_rows
    cols = 28 - 2 * empty_cols

    max_image_count = 0 if empty_rows == 0 and empty_cols == 0 else\
                      2 * empty_rows + 2 * empty_cols if empty_rows == 0 or empty_cols == 0 else\
                      4 * empty_rows * empty_cols

    image_count = math.ceil(max_image_count * density)
    samples = [(sample, 0, 0)]

    for i in range(image_count):
        shift_row = 0 if empty_rows == 0 else random.randint(-empty_rows, empty_rows)
        shift_col = 0 if empty_cols == 0 else random.randint(-empty_cols, empty_cols)
        from_row = empty_rows + shift_row
        from_col = empty_cols + shift_col
        shifted_sample = np.zeros((28, 28))
        shifted_sample[from_row:from_row+rows, from_col:from_col+cols] = sub_image
        samples.append((shifted_sample, shift_row, shift_col))

    return samples


assert x.shape == (392, 28, 28, 1)
x_list = []
y_list = []

for i in range(m):
    x_sample = [x[[i], ...]]
    encoding = autoencoder.predict(x_sample)
    assert encoding.shape == (1, 1, 1, 14)
    encoding = encoding.reshape((1, 1, 14))
    # print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), encoding[0, 0, :]))))

    x_sample = x_sample[0][0, :, :, 0]
    assert x_sample.shape == (28, 28)

    samples = generated_shifted_samples(x_sample, density=0.1)
    for sample, shift_row, shift_col in samples:
        assert sample.shape == (28, 28)
        sample = sample.reshape((28, 28, 1))
        shift = np.zeros((1, 1, 1, 2))
        shift[0, 0, 0, :] = [shift_col/27, shift_row/27]  # 28 points, but 27 ranges!!!
        shift = d + shift
        y = np.zeros((28, 28, 17))
        y[:, :, [0]] = sample
        y[:, :, 1:3] = sample * shift
        y[:, :, 3:17] = sample * encoding
        y_list.append(y.reshape((1, 28, 28, 17)))
        x_list.append(sample.reshape((1, 28, 28, 1)))

    print("done", i, "with", len(samples), "samples")


X = np.concatenate(x_list)
print('sample shape:', X.shape)
assert X.shape[1:] == (28, 28, 1)

Y = np.concatenate(y_list)
print('encoding shape:', Y.shape)
assert Y.shape[1:] == (28, 28, 17)

np.save('line_samples_v2_{}x28x28x1.npy'.format(X.shape[0]), X)
np.save('line_encodings_v2_{}x28x28x16.npy'.format(Y.shape[0]), Y)

print('done all')
