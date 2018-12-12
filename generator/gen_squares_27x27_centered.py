import time as t
import numpy as np
import libs.utilities as utl
import libs.generators as gens
from libs.concepts import Concept


def square_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params

    # 0-based 2,4  4,25  25,23  23,2
    # x,y  y,w-x-1  w-x-1,w-y-1  w-y-1,x
    draw.line((x, y, y, dim - 1 - x), fill=255)
    draw.line((y, dim - 1 - x, dim - x - 1, dim - y - 1), fill=255)
    draw.line((dim - x - 1, dim - y - 1, dim - y - 1, x), fill=255)
    draw.line((dim - y - 1, x, x, y), fill=255)


def get_images(dim=27):
    quadrant_width = dim // 2 - 1

    image_list = []
    for x in range(quadrant_width):
        for y in range(quadrant_width):
            params = [x, y]
            image = gens.draw_image(dim, params, drawer=square_drawer, show=False)
            image = image.astype('float32') / 255.
            image_list.append(image)

    image_list = np.array(image_list)

    return image_list, len(image_list)


def main():

    dim = 27
    vertices = 4
    regions = 9
    channels_full = 17
    cluster_threshold = 2
    adjacency_threshold = -30
    embedding_threshold = 0.9

    saving = True
    testing = False

    image_list, m = get_images(dim=dim)
    region_list = utl.get_regions(regions)

    concept = Concept.LINE

    _, _, decoder_model = concept.model_autoencoder()
    matrix_encoder_model = concept.model_matrix_encoder()
    clustering_model = concept.model_clustering()

    if testing:

        test_index = 165  # 166  # 142  # 127  # 100  #73  # 26
        image = image_list[test_index]

        vertex_list, edge_list = gens.get_graph(
            decoder_model, matrix_encoder_model, clustering_model, image,
            adjacency_threshold, embedding_threshold, cluster_threshold, dim=dim, show=True)

        print('image no:', test_index)
        print('\tvertex_list:', len(vertex_list))
        print('\tedge_list:', len(edge_list), edge_list)

        return

    vertex_matrix = []
    edge_matrix = []
    mapping_matrix = []

    for index, image in enumerate(image_list):

        vertex_list, edge_list = gens.get_graph(
            decoder_model, matrix_encoder_model, clustering_model, image,
            adjacency_threshold, embedding_threshold, cluster_threshold, dim=dim)

        # only accept graphs with predefined number of vertices
        if len(vertex_list) != vertices or len(edge_list) != vertices:
            continue

        print('image no:', index)

        # preparing data for mapping matrix generation
        adjacency_matrix = utl.get_adjacency_matrix_from_edges(vertices, edge_list)
        region_matrix = utl.get_region_matrix(vertex_list, region_list, show=False, debug=False)
        row_indices, column_indices, node_indices = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

        # embed node indices at calculated locations in mapping matrix
        mapping_list = np.full((vertices, regions), -1)
        mapping_list[row_indices, column_indices] = node_indices

        vertex_matrix.extend(vertex_list)
        edge_matrix.extend(edge_list)
        mapping_matrix.extend(mapping_list)

    vertex_matrix = np.array(vertex_matrix).reshape((-1, vertices, channels_full))
    edge_matrix = np.array(edge_matrix).reshape((-1, vertices, 2))
    mapping_matrix = np.array(mapping_matrix).reshape((-1, vertices, regions))

    m = len(vertex_matrix)
    assert vertex_matrix.shape == (m, vertices, channels_full)
    assert edge_matrix.shape == (m, vertices, 2)
    assert mapping_matrix.shape == (m, vertices, regions)

    if saving:
        timestamp = int(t.time())
        prefix = 'data/square/square-centered'

        np.save('{}-images-{}-{}x{}x{}.npy'.format(prefix, timestamp, m, dim, dim), image_list)
        np.save('{}-vertices-{}-{}x{}x{}.npy'.format(prefix, timestamp, m, vertices, channels_full), vertex_matrix)
        np.save('{}-edges-{}-{}x{}x2.npy'.format(prefix, timestamp, m, vertices), edge_matrix)
        np.save('{}-mappings-{}-{}x{}x{}.npy'.format(prefix, timestamp, m, vertices, regions), mapping_matrix)


if __name__ == '__main__':
    main()
    print('end')
