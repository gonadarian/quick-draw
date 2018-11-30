import numpy as np
import random as rand
import models as mdls
import utilities as utl
import matplotlib.pyplot as plt


random_sample = False
sample_id = 105121
analysis = False
saving = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def load_quick_draw_data_set():
    # TODO use datasets load method
    data = np.load('generator/data/quick_draw-square.npy')
    data = data.reshape((-1, 28, 28))
    data = data.astype('float32') / 255.
    return data


def main():

    data_set = load_quick_draw_data_set()
    index = rand.randint(0, len(data_set)) if random_sample else sample_id
    print('using quick draw sample', index)
    sample = data_set[index, :, :]

    assert sample.shape == (28, 28)

    decoder_model = mdls.load_decoder_model()
    encoder_model = mdls.load_encoder_model()
    clustering_model = mdls.load_clustering_model()

    embeddings = utl.get_embeddings(encoder_model, sample, threshold=0.9, show=False)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    clusters = utl.extract_clusters(cluster_matrix)

    if analysis:
        utl.show_elbow_curve(embeddings, True)
        plt.imshow(cluster_matrix)
        plt.show()

    images = []
    nodes = []
    for cluster in clusters:
        if len(cluster) > 2:
            cluster_embeddings = embeddings[list(cluster)]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            encoding, center = utl.extract_encoding_and_center(cluster_embedding)
            assert encoding.shape == (14, )
            assert center.shape == (2, )
            image = utl.gen_image(decoder_model, encoding, center, show=False)
            images.append(image)
            nodes.append(cluster_embedding)

    utl.show_clusters(sample, images)

    adjacency_matrix = utl.get_adjacency_matrix(images, show=False)
    adjacency_matrix = adjacency_matrix > -30
    print(adjacency_matrix)

    edges = utl.get_graph_edges(adjacency_matrix)
    utl.draw_graph(edges)

    nodes_count = len(nodes)
    region_count = 8
    encoding_dim = 17

    edges = np.array(edges)
    nodes = np.array(nodes)

    regions = utl.get_regions(region_count=region_count, show=True)
    adjacency_matrix = utl.get_adjacency_matrix_from_edges(nodes_count, edges)
    region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

    nodes = nodes
    row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

    mappings = np.full((nodes_count, region_count+1), -1)
    mappings[row_indexes, column_indexes] = node_indexes

    mappings = mappings.reshape((1, nodes_count, region_count+1))
    nodes = nodes.reshape((1, nodes_count, encoding_dim))

    autoencoder_model = mdls.load_graph_autoencoder_model(node_count=4, region_count=9, version=4)
    decoded_nodes = autoencoder_model.predict(x=[nodes, mappings])
    decoded_nodes = decoded_nodes[0, :, :]
    print(decoded_nodes)

    images = []
    for node in decoded_nodes:
        encoding, center = utl.extract_encoding_and_center(node)
        assert encoding.shape == (14, )
        assert center.shape == (2, )
        image = utl.gen_image(decoder_model, encoding, center, show=False)
        images.append(image)

    utl.show_clusters(sample, images)


if __name__ == '__main__':
    main()
    print('end')
