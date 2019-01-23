import numpy as np
import random as rand
import libs.models as mdl
import libs.datasets as ds
import libs.utilities as utl
import libs.clustering as clst
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from libs.concepts import Concept


dim = 27
vertices = 4
channels_full = 17
channels = 14

cluster_threshold = 3
adjacency_threshold = -30
similarity_threshold = 0.8

mnist_data = True
mnist_sample = None  # 5514/9
mnist_category = None
quickdraw_data = False
quickdraw_sample = None  # 64935  # 99209  # 115331
quickdraw_category = 'triangle'
custom_data = False
custom_sample = 3

prediction = 'all'
analysis = False
saving = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def load_quickdraw_dataset():

    # good choices of categories for different concepts:
    # - LINE: triangle, square, octagon
    # - ELLIPSE: circle
    # - BEZIER: animal_migration, rainbow, moon

    x = ds.load_images_quickdraw(quickdraw_category, dim=27)

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


def get_sample():

    if mnist_data:
        category = rand.randint(0, 9) if mnist_category is None else mnist_category
        data_set = ds.load_images_mnist(category=category, dim=dim)
        index = rand.randint(0, len(data_set)) if mnist_sample is None else mnist_sample
        print('using mnist sample', index)
        sample = data_set[index, :, :]

    elif quickdraw_data:
        data_set = load_quickdraw_dataset()  # concept)
        index = rand.randint(0, len(data_set)) if quickdraw_sample is None else quickdraw_sample
        print('using quick draw sample', index)
        sample = data_set[index, :, :]

    else:
        sample = load_custom_data_sample(shape=custom_sample, show=False)

    assert sample.shape == (dim, dim)

    thinner_model = mdl.load_matrix_thinner_mix_model()
    sample = thinned_sample(thinner_model, sample)
    assert sample.shape == (dim, dim)

    return sample


def predict(concept, sample):

    matrix_encoder_model = concept.get_model_matrix_encoder()
    embeddings = utl.get_embeddings(matrix_encoder_model, sample, dim=dim, threshold=0.8, show=False)
    print('[predict] embeddings:', embeddings.shape)

    predict_from_embeddings(concept, sample, embeddings)


def predict_from_embeddings(concept, sample, embeddings):

    if len(embeddings) == 0:
        print('[predict_from_embeddings] WARNING: no embeddings provided for concept', concept)
        return

    _, _, decoder_model = concept.get_model_autoencoder()
    clustering_model = concept.get_model_clustering()

    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    print('[predict_from_embeddings] cluster_matrix:', cluster_matrix.shape)

    # clusters = utl.extract_clusters_v1(cluster_matrix)
    _, clusters = clst.cluster(embeddings, concept, similarity_threshold=similarity_threshold, show=False)
    print('[predict_from_embeddings] clusters:', len(clusters), [len(cluster) for cluster in clusters])

    if analysis:
        plt.imshow(cluster_matrix)
        plt.show()

    lines = []
    images = []
    weights = []

    for cluster in filter(lambda it: len(it) > cluster_threshold, clusters):

        cluster_embeddings = embeddings[list(cluster)]
        cluster_embedding = np.mean(cluster_embeddings, axis=0)
        encoding, center = utl.extract_encoding_and_center(cluster_embedding)
        assert encoding.shape == (channels, )
        assert center.shape == (2, )

        image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
        lines.append(cluster_embedding)
        images.append(image)
        weights.append(len(cluster_embeddings))

    utl.show_clusters(sample, images, weights, dim=dim)

    adjacency_matrix = utl.get_adjacency_matrix(images, dim=dim, show=True)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    edges = utl.get_graph_edges(adjacency_matrix)

    if analysis:
        print(adjacency_matrix)
        utl.draw_graph(edges)


def predict_single(sample, concept):

    sample_threshold = 0.8
    matrix_encoder_model = concept.get_model_matrix_encoder()
    embeddings = utl.get_embeddings(matrix_encoder_model, sample, threshold=sample_threshold)
    print('[predict_single] embeddings:', embeddings.shape)

    similarities = []
    decoder_model = concept.get_model_autoencoder()[2]

    for i, embedding in enumerate(embeddings):
        encoding, center = utl.extract_encoding_and_center(embedding)
        decoded_image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
        similarity = utl.calc_similarity(sample, decoded_image, algorithm='gauss')
        similarities.append(similarity)

    return embeddings, similarities


def predict_all(sample):

    line_data = predict_single(sample, Concept.LINE)
    ellipse_data = predict_single(sample, Concept.ELLIPSE)
    bezier_data = predict_single(sample, Concept.BEZIER)

    similarities = np.vstack((line_data[1], ellipse_data[1], bezier_data[1]))
    print('[predict_all] similarities:', similarities.shape)

    choices = np.argmax(similarities, axis=0)
    print('[predict_all] choices:', choices.shape, '\n', choices)

    lines = line_data[0][choices == 0]
    ellipses = ellipse_data[0][choices == 1]
    beziers = bezier_data[0][choices == 2]

    print('[predict_all] lines   :', lines.shape)
    print('[predict_all] ellipses:', ellipses.shape)
    print('[predict_all] beziers :', beziers.shape)

    predict_from_embeddings(Concept.LINE, sample, lines)
    predict_from_embeddings(Concept.ELLIPSE, sample, ellipses)
    predict_from_embeddings(Concept.BEZIER, sample, beziers)


def main():
    sample = get_sample()

    if prediction == 'all':
        predict_all(sample)

    else:
        predict(Concept.LINE, sample)
        predict(Concept.ELLIPSE, sample)
        predict(Concept.BEZIER, sample)
        predict(Concept.STAR, sample)


if __name__ == '__main__':
    while True:
        main()
