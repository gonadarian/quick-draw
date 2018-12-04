import numpy as np
import random as rand
import models as mdls
import utilities as utl
import matplotlib.pyplot as plt


dim = 28
embedding_threshold = 0.9
cluster_threshold = 2
adjacency_threshold = -30
channels_full = 17
channels = 14
regions = 9
vertices = 4

random_sample = False
sample_id = 105121
analysis = False
saving = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def load_quick_draw_data_set():
    # TODO use datasets load method
    data = np.load('generator/data/quick_draw-square.npy')
    data = data.reshape((-1, dim, dim))
    data = data.astype('float32') / 255.

    return data


def main():

    data_set = load_quick_draw_data_set()
    index = rand.randint(0, len(data_set)) if random_sample else sample_id
    print('using quick draw sample', index)
    sample = data_set[index, :, :]

    assert sample.shape == (dim, dim)

    decoder_model = mdls.load_decoder_model()
    encoder_model = mdls.load_encoder_model()
    clustering_model = mdls.load_clustering_model()

    embeddings = utl.get_embeddings(encoder_model, sample, threshold=embedding_threshold, show=False)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    clusters = utl.extract_clusters(cluster_matrix)

    if analysis:
        utl.show_elbow_curve(embeddings, True)
        plt.imshow(cluster_matrix)
        plt.show()

    images = []
    nodes = []
    for cluster in clusters:
        if len(cluster) > cluster_threshold:
            cluster_embeddings = embeddings[list(cluster)]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            encoding, center = utl.extract_encoding_and_center(cluster_embedding)
            assert encoding.shape == (channels, )
            assert center.shape == (2, )
            image = utl.gen_image(decoder_model, encoding, center, show=False)
            images.append(image)
            nodes.append(cluster_embedding)

    utl.show_clusters(sample, images)

    adjacency_matrix = utl.get_adjacency_matrix(images, show=False)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    print(adjacency_matrix)

    edges = utl.get_graph_edges(adjacency_matrix)
    utl.draw_graph(edges)

    nodes_count = len(nodes)
    encoding_dim = channels_full

    edges = np.array(edges)
    nodes = np.array(nodes)

    adjacency_matrix = utl.get_adjacency_matrix_from_edges(nodes_count, edges)
    region_list = utl.get_regions(regions)
    region_matrix = utl.get_region_matrix(nodes, region_list, show=True, debug=True)

    row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

    mappings = np.full((nodes_count, regions), -1)
    mappings[row_indexes, column_indexes] = node_indexes

    mappings = mappings.reshape((1, nodes_count, regions))
    nodes = nodes.reshape((1, nodes_count, encoding_dim))

    autoencoder_model = mdls.load_graph_autoencoder_model(vertices, regions, version=4)
    decoded_nodes = autoencoder_model.predict(x=[nodes, mappings])
    decoded_nodes = decoded_nodes[0, :, :]
    print(decoded_nodes)

    images = []
    for node in decoded_nodes:
        encoding, center = utl.extract_encoding_and_center(node)
        assert encoding.shape == (channels, )
        assert center.shape == (2, )
        image = utl.gen_image(decoder_model, encoding, center, show=False)
        images.append(image)

    utl.show_clusters(sample, images)


if __name__ == '__main__':
    main()
    print('end')
