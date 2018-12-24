import math
import numpy as np
import random as rand
import aggdraw as agg
import libs.utilities as utl
from PIL import Image, ImageDraw


def show_image(image):
    image = Image.fromarray(image)
    image.show()
    return


def draw_image(dim, params, drawer, rotate=None, antialias=False, show=False):
    image = Image.new("L", (dim, dim), "black")

    draw = agg.Draw(image)
    if not antialias:
        draw.setantialias(False)
    pen = agg.Pen("white", 1)  # 5 is the outline width in pixels

    # draw = ImageDraw.Draw(image)
    drawer(draw, dim, params, pen)
    draw.flush()

    if rotate:
        resample = Image.CUBIC if antialias else Image.NEAREST
        rotated = image.rotate(rotate, resample=resample, expand=False)
        image = Image.new('L', (dim, dim), 'black')
        image.paste(rotated, (0, 0), rotated)

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

    [empty_rows, empty_cols] = np.amin(np.where(sample > 0), axis=1)
    sub_image = sample[empty_rows:dim - empty_rows, empty_cols:dim - empty_cols]

    # lines are not centered after all.... so +1/-1 on couple of places :(
    rows = dim - 2 * empty_rows
    cols = dim - 2 * empty_cols

    max_image_count = (0 if empty_rows == 0 and empty_cols == 0 else
                       2 * empty_rows + 2 * empty_cols if empty_rows == 0 or empty_cols == 0 else
                       4 * empty_rows * empty_cols)

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


def get_graph(decoder_model, encoder_model, clustering_model, sample,
              adjacency_threshold, embedding_threshold, cluster_threshold, dim=27, show=False):

    # TODO embeddings and clusters should be done in a loop, individually per concept
    embedding_list = utl.get_embeddings(encoder_model, sample, dim=dim, threshold=embedding_threshold, show=False)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embedding_list)
    cluster_list = utl.extract_clusters(cluster_matrix)

    image_list = []
    vertex_list = []

    for cluster in cluster_list:
        if len(cluster) > cluster_threshold:

            cluster_embeddings = embedding_list[list(cluster)]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            encoding, center = utl.extract_encoding_and_center(cluster_embedding)
            assert len(encoding.shape) == 1
            assert center.shape == (2, )

            image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
            image_list.append(image)
            vertex_list.append(cluster_embedding)

    # TODO adjacency matrix should be calculated against all supported concepts
    adjacency_matrix = utl.get_adjacency_matrix(image_list, dim=dim, show=False)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    edge_list = utl.get_graph_edges(adjacency_matrix)

    if show:
        utl.show_clusters(sample, image_list, dim=dim)
        print(adjacency_matrix)
        utl.draw_graph(edge_list)

    return np.array(vertex_list), np.array(edge_list)


def calc_image_center(image):
    indices = np.argwhere(image > 0)
    m = len(indices)

    pixels = image[indices[:, 0], indices[:, 1]].reshape(m, 1)
    pixels_sum = np.sum(pixels)

    weighted = np.multiply(pixels, indices)
    weighted = np.sum(weighted, axis=0)
    weighted = weighted / pixels_sum
    weighted = np.rint(weighted).astype(np.int64)

    return weighted
