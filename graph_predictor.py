import numpy as np
import random as rand
import libs.models as mdls
import libs.datasets as ds
import libs.utilities as utl
import matplotlib.pyplot as plt
from libs.concepts import Concept


dim = 27
embedding_threshold = 0.9
cluster_threshold = 2
adjacency_threshold = -30
channels_full = 17
channels = 14
regions = 9
vertices = 4

random_sample = True
sample_index = 65165
analysis = False
saving = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


def main():
    concept = Concept.LINE
    quickdraw_category = 'square'

    dataset = ds.load_images_quickdraw(quickdraw_category, dim=dim)
    index = rand.randint(0, len(dataset)) if random_sample else sample_index
    print('using {} quickdraw sample: {}'.format(quickdraw_category, index))
    sample = dataset[index, :, :]

    assert sample.shape == (dim, dim)

    _, _, decoder_model = concept.model_autoencoder()
    matrix_encoder_model = concept.model_matrix_encoder()
    clustering_model = concept.model_clustering()

    embeddings = utl.get_embeddings(matrix_encoder_model, sample, dim=dim, threshold=embedding_threshold, show=False)
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

            image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
            images.append(image)
            nodes.append(cluster_embedding)

    utl.show_clusters(sample, images, dim=dim)

    adjacency_matrix = utl.get_adjacency_matrix(images, dim=dim, show=False)
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

    autoencoder_model = mdls.load_graph_autoencoder_model()
    decoded_nodes = autoencoder_model.predict(x=[nodes, mappings])
    decoded_nodes = decoded_nodes[0, :, :]
    print(decoded_nodes)

    images = []
    for node in decoded_nodes:
        encoding, center = utl.extract_encoding_and_center(node)
        assert encoding.shape == (channels, )
        assert center.shape == (2, )
        image = utl.gen_image(decoder_model, encoding, center, dim=dim, show=False)
        images.append(image)

    utl.show_clusters(sample, images, dim=dim)


if __name__ == '__main__':
    main()
    print('end')
