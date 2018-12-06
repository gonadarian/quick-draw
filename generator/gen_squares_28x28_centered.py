import numpy as np
import libs.models as mdls
import libs.utilities as utl
import libs.generators as gens


vertices = 4
edges = 4
regions = 9
channels = 14
channels_full = 17
adjacency_threshold = -30

saving = True
testing = False


def square_drawer(draw, dim, params):
    assert len(params) == 2
    x, y = params
    # 0-based 2,4  4,25  25,23  23,2
    # x,y  y,w-x-1  w-x-1,w-y-1  w-y-1,x
    draw.line((x, y, y, dim - 1 - x), fill=255)
    draw.line((y, dim - 1 - x, dim - x - 1, dim - y - 1), fill=255)
    draw.line((dim - x - 1, dim - y - 1, dim - y - 1, x), fill=255)
    draw.line((dim - y - 1, x, x, y), fill=255)


def get_images():
    dim = 28
    quadrant_width = dim // 2 - 1

    image_list = []
    for x in range(quadrant_width):
        for y in range(quadrant_width):
            params = [x, y]
            image = gens.draw_image(dim, params, drawer=square_drawer, show=False)
            image = image.astype('float32') / 255.
            image_list.append(image)

    return np.array(image_list)


def get_graph(decoder_model, encoder_model, clustering_model, sample, show=False):
    embedding_list = utl.get_embeddings(encoder_model, sample, threshold=0.9, show=False)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embedding_list)
    cluster_list = utl.extract_clusters(cluster_matrix)

    image_list = []
    vertex_list = []
    for cluster in cluster_list:
        if len(cluster) > 2:
            cluster_embeddings = embedding_list[list(cluster)]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            encoding, center = utl.extract_encoding_and_center(cluster_embedding)
            assert encoding.shape == (channels, )
            assert center.shape == (2, )

            image = utl.gen_image(decoder_model, encoding, center, show=False)
            image_list.append(image)
            vertex_list.append(cluster_embedding)

    adjacency_matrix = utl.get_adjacency_matrix(image_list, show=False)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    edge_list = utl.get_graph_edges(adjacency_matrix)

    if show:
        utl.show_clusters(sample, image_list)
        print(adjacency_matrix)
        utl.draw_graph(edge_list)

    return vertex_list, edge_list


def main():
    image_list = get_images()
    m = len(image_list)

    if saving:
        np.save('data\square_originals_v1_{}x28x28.npy'.format(m), image_list)

    decoder_model = mdls.load_decoder_model()
    encoder_model = mdls.load_encoder_model()
    clustering_model = mdls.load_clustering_model()

    if testing:
        test_index = 165  # 166  # 142  # 127  # 100  #73  # 26
        image = image_list[test_index]
        vertex_list, edge_list = get_graph(decoder_model, encoder_model, clustering_model, image, show=True)

        print('image no:', test_index)
        print('\tlines:', len(vertex_list))
        print('\tedge_list:', len(edge_list), edge_list)

        return

    vertex_matrix = []
    edge_matrix = []

    # TODO convert to enumerate
    for index in range(m):
        image = image_list[index]
        vertex_list, edge_list = get_graph(decoder_model, encoder_model, clustering_model, image)

        if len(vertex_list) == 4 and len(edge_list) == 4:
            print('image no:', index)
            vertex_matrix.extend(vertex_list)
            edge_matrix.extend(edge_list)

    vertex_matrix = np.array(vertex_matrix).reshape((-1, 4, channels_full))
    edge_matrix = np.array(edge_matrix).reshape((-1, 4, 2))

    m = len(vertex_matrix)
    assert vertex_matrix.shape == (m, 4, channels_full)
    assert edge_matrix.shape == (m, 4, 2)

    if saving:
        np.save('data/graph_lines_set_v1_{}x4x{}.npy'.format(m, channels_full), vertex_matrix)
        np.save('data/graph_edges_set_v1_{}x4x2.npy'.format(m), edge_matrix)

    mapping_set = []
    region_list = utl.get_regions(regions)

    for index in range(m):
        vertex_list = vertex_matrix[index]
        edge_list = edge_matrix[index]

        adjacency_matrix = utl.get_adjacency_matrix_from_edges(vertices, edge_list)
        region_matrix = utl.get_region_matrix(vertex_list, region_list, show=True, debug=True)

        row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

        mapping = np.full((vertices, regions), -1)
        mapping[row_indexes, column_indexes] = node_indexes

        mapping_set.extend(mapping)

    mapping_set = np.array(mapping_set).reshape((-1, 4, regions))
    assert mapping_set.shape == (m, 4, regions)

    if saving:
        np.save('data/graph_mapping_set_v1_{}x4x{}.npy'.format(m, regions), mapping_set)


if __name__ == '__main__':
    main()
    print('end')
