import numpy as np
import models as mdls
import utilities as utl
from PIL import Image, ImageDraw


w = 28
node_count = 4
edge_count = 4
region_count = 9
channels = 14
channels_full = 17
adjacency_threshold = -30

saving = True
testing = False


def get_img_array(x, y, show=False):
    image = Image.new("L", (w, w), "black")

    draw = ImageDraw.Draw(image)

    # 0-based 2,4  4,25  25,23  23,2
    # x,y  y,w-x-1  w-x-1,w-y-1  w-y-1,x
    draw.line((x, y, y, w-1-x), fill=255)
    draw.line((y, w-1-x, w-x-1, w-y-1), fill=255)
    draw.line((w-x-1, w-y-1, w-y-1, x), fill=255)
    draw.line((w-y-1, x, x, y), fill=255)

    image_array = np.asarray(image)

    if show:
        print(image_array)
        image.show()

    return image_array


def get_images():
    quadrant_w = w // 2 - 1
    im_set = np.empty([quadrant_w ** 2, w, w])

    for x in range(quadrant_w):
        for y in range(quadrant_w):
            im_array = get_img_array(x, y, False)
            im_array = im_array.astype('float32') / 255.
            im_set[x * quadrant_w + y, :, :] = im_array

    return im_set


def get_graph(decoder_model, encoder_model, clustering_model, sample, show=False):
    embeddings = utl.get_embeddings(encoder_model, sample, threshold=0.9, show=False)
    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings)
    clusters = utl.extract_clusters(cluster_matrix)

    images = []
    lines = []
    for cluster in clusters:
        if len(cluster) > 2:
            cluster_embeddings = embeddings[list(cluster)]
            cluster_embedding = np.mean(cluster_embeddings, axis=0)
            encoding, center = utl.extract_encoding_and_center(cluster_embedding)
            assert encoding.shape == (channels, )
            assert center.shape == (2, )
            image = utl.gen_image(decoder_model, encoding, center, show=False)
            images.append(image)
            lines.append(cluster_embedding)

    adjacency_matrix = utl.get_adjacency_matrix(images, show=False)
    adjacency_matrix = adjacency_matrix > adjacency_threshold
    edges = utl.get_graph_edges(adjacency_matrix)

    if show:
        utl.show_clusters(sample, images)
        print(adjacency_matrix)
        utl.draw_graph(edges)

    return lines, edges


def main():
    images = get_images()

    if saving:
        np.save('data\square_originals_v1_{}x28x28.npy'.format(len(images)), images)

    decoder_model = mdls.load_decoder_model()
    encoder_model = mdls.load_encoder_model()
    clustering_model = mdls.load_clustering_model()

    if testing:

        test_index = 165  # 166  # 142  # 127  # 100  #73  # 26
        nodes, edges = get_graph(decoder_model, encoder_model, clustering_model, images[test_index], show=True)
        print('image no:', test_index)
        print('\tlines:', len(nodes))
        print('\tedges:', len(edges), edges)

        return

    nodes_set = []
    edges_set = []

    for index in range(len(images)):
        image = images[index]
        nodes, edges = get_graph(decoder_model, encoder_model, clustering_model, image)
        if len(nodes) == 4 and len(edges) == 4:
            print('image no:', index)
            nodes_set.extend(nodes)
            edges_set.extend(edges)

    nodes_set = np.array(nodes_set).reshape((-1, 4, channels_full))
    edges_set = np.array(edges_set).reshape((-1, 4, 2))

    m = len(nodes_set)
    assert nodes_set.shape == (m, 4, channels_full)
    assert edges_set.shape == (m, 4, 2)

    if saving:
        np.save('data/graph_lines_set_v1_{}x4x{}.npy'.format(m, channels_full), nodes_set)
        np.save('data/graph_edges_set_v1_{}x4x2.npy'.format(m), edges_set)

    mapping_set = []
    regions = utl.get_regions(region_count)

    for index in range(m):
        nodes = nodes_set[index]
        edges = edges_set[index]

        adjacency_matrix = utl.get_adjacency_matrix_from_edges(node_count, edges)
        region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

        row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

        mapping = np.full((node_count, region_count), -1)
        mapping[row_indexes, column_indexes] = node_indexes

        mapping_set.extend(mapping)

    mapping_set = np.array(mapping_set).reshape((-1, 4, region_count))
    assert mapping_set.shape == (m, 4, region_count)

    if saving:
        np.save('data/graph_mapping_set_v1_{}x4x{}.npy'.format(m, region_count), mapping_set)


if __name__ == '__main__':
    main()
    print('end')
