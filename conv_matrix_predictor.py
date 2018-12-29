import numpy as np
import random as rand
import libs.models as mdl
import libs.datasets as ds
import libs.utilities as utl
import matplotlib.pyplot as plt
from libs.concepts import Concept
from PIL import Image, ImageDraw


dim = 27
vertices = 4
cluster_threshold = 3
adjacency_threshold = -30
channels_full = 17
channels = 14

mnist = True
mnist_category = 3
mnist_sample = None  # 1359  # 5129  # 2583
quickdraw_data = True
quickdraw_sample = None
custom_sample = 3
analysis = False
saving = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def load_quickdraw_dataset(concept):
    switcher = {
        Concept.LINE: 'octagon',  # options: triangle, square, octagon
        Concept.ELLIPSE: 'circle',  # options: circle
        Concept.BEZIER: 'moon',  # options: animal_migration, rainbow, moon
    }

    category = switcher.get(concept)
    x = ds.load_images_quickdraw(category, dim=27)

    return x


def load_custom_data_sample(shape=1, show=False):
    im = Image.new("L", (dim, dim), "black")
    draw = ImageDraw.Draw(im)

    if shape == 0:  # shape: square
        draw.line((5,  5, 5,  22), fill=255)
        draw.line((5,  5, 22, 5), fill=255)
        draw.line((22, 5, 22, 22), fill=255)
        draw.line((5, 22, 22, 22), fill=255)

    if shape == 1:  # shape: rhomboid
        draw.line((5,  14, 14, 5), fill=255)
        draw.line((14, 5,  23, 14), fill=255)
        draw.line((23, 14, 14, 23), fill=255)
        draw.line((14, 23, 5,  14), fill=255)

    if shape == 2:  # shape: tilted, parallel lines
        draw.line((2, 2,  25, 6), fill=255)
        draw.line((2, 8,  25, 12), fill=255)
        draw.line((2, 14, 25, 18), fill=255)
        draw.line((2, 20, 25, 24), fill=255)

    if shape == 3:  # shape: star
        draw.line((5,  10, 23, 18), fill=255)
        draw.line((10, 5,  18, 23), fill=255)
        draw.line((18, 5,  10, 23), fill=255)
        draw.line((23, 10, 5,  18), fill=255)

    data = np.asarray(im)
    if show:
        im.show()

    data = data.astype('float32') / 255.
    return data


def thinned_sample(thinner_model, sample):
    sample = sample.reshape((1, dim, dim, 1))
    thinned = thinner_model.predict(sample)

    return thinned.reshape(dim, dim)


def main(concept):

    if mnist:
        data_set = ds.load_images_mnist(category=mnist_category, dim=dim)
        index = rand.randint(0, len(data_set)) if mnist_sample is None else mnist_sample
        print('using mnist sample', index)
        sample = data_set[index, :, :]

    elif quickdraw_data:
        data_set = load_quickdraw_dataset(concept)
        index = rand.randint(0, len(data_set)) if quickdraw_sample is None else quickdraw_sample
        print('using quick draw sample', index)
        sample = data_set[index, :, :]

    else:
        sample = load_custom_data_sample(shape=custom_sample, show=False)

    assert sample.shape == (dim, dim)

    thinner_model = mdl.load_matrix_thinner_mix_model()
    sample = thinned_sample(thinner_model, sample)
    assert sample.shape == (dim, dim)

    _, _, decoder_model = concept.model_autoencoder()
    matrix_encoder_model = concept.model_matrix_encoder()
    clustering_model = concept.model_clustering()
    sample_threshold = 0.8

    embeddings = utl.get_embeddings(matrix_encoder_model, sample, dim=dim, threshold=sample_threshold, show=False)
    print('embeddings:', embeddings.shape)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    print('cluster_matrix:', cluster_matrix.shape)
    clusters = utl.extract_clusters(cluster_matrix)
    print('clusters:', len(clusters), [len(cluster) for cluster in clusters])

    if analysis:
        utl.show_elbow_curve(embeddings, True)
        plt.imshow(cluster_matrix)
        plt.show()

    images = []
    lines = []
    for cluster in clusters:
        if len(cluster) <= cluster_threshold:
            continue

        cluster_embeddings = embeddings[list(cluster)]
        cluster_embedding = np.mean(cluster_embeddings, axis=0)
        encoding, center = utl.extract_encoding_and_center(cluster_embedding)
        assert encoding.shape == (channels, )
        assert center.shape == (2, )
        image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
        images.append(image)
        lines.append(cluster_embedding)

    utl.show_clusters(sample, images, dim=dim)

    adjacency_matrix = utl.get_adjacency_matrix(images, dim=dim, show=True)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    print(adjacency_matrix)

    edges = utl.get_graph_edges(adjacency_matrix)
    utl.draw_graph(edges)

    if saving:
        print(lines)
        np.save('generator/data/graph_vertices.npy', np.array(lines))
        print(edges)
        np.save('generator/data/graph_edges.npy', np.array(edges))


if __name__ == '__main__':

    # main(Concept.LINE)
    # main(Concept.ELLIPSE)
    main(Concept.BEZIER)

    print('end')
