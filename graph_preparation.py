import numpy as np
import libs.datasets as ds
import libs.utilities as utl


vertices = 4
channels_full = 17
region_count = 9

real_data = True
show = False


def main():

    nodes = (ds.load_graph_lines() if real_data else
             np.arange(vertices * channels_full).reshape((vertices, channels_full)))
    assert nodes.shape == (vertices, channels_full)

    edges = ds.load_graph_edges()
    assert edges.shape == (vertices, 2)

    if show:
        print('nodes:\n', nodes)
        print('edges:\n', edges)

    regions = utl.get_regions(region_count)

    adjacency_matrix = utl.get_adjacency_matrix_from_edges(vertices, edges)
    region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

    if show:
        print('adjacency_matrix:\n', adjacency_matrix)
        print('region_matrix:\n', region_matrix)

    # matrix approach
    row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)
    # max 1-neighbourhood size is 8, plus 1 for center node
    matrix_v1 = np.zeros((vertices, region_count, channels_full))
    # fill in the values
    matrix_v1[row_indexes, column_indexes, :] = nodes[node_indexes]

    # vector approach
    vector_indexes, node_indexes = utl.get_vector_transformation(adjacency_matrix, region_matrix)
    # max 1-neighbourhood size is 8, plus 1 for center node
    vector = np.zeros((vertices * region_count, channels_full))
    # fill in the values
    vector[vector_indexes, :] = nodes[node_indexes]
    matrix_v2 = vector.reshape((vertices, region_count, channels_full))

    if show:
        print('matrix_v1:\n', matrix_v1)
        print('matrix_v2:\n', matrix_v2)


if __name__ == '__main__':
    main()
    print('end')
