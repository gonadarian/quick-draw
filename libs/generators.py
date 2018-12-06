import math
import numpy as np
import random as rand
from PIL import Image, ImageDraw


def draw_image(dim, params, drawer, show=False):
    image = Image.new("L", (dim, dim), "black")

    draw = ImageDraw.Draw(image)
    drawer(draw, dim, params)
    image_array = np.asarray(image)

    if show:
        print(image_array)
        image.show()

    return image_array


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


def generated_shifted_samples(sample, dim, density=0.3):
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
