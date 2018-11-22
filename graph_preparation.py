import numpy as np
import datasets as ds
import utilities as utl


dim = 4
real_data = True
show = False


def main():

    nodes = ds.load_graph_lines() if real_data else np.arange(dim * 17).reshape((dim, 17))
    assert nodes.shape == (dim, 17)

    edges = ds.load_graph_edges()
    assert edges.shape == (dim, 2)

    if show:
        print('nodes:\n', nodes)
        print('edges:\n', edges)

    regions = utl.get_regions(region_count=8, show=True)

    adjacency_matrix = utl.get_adjacency_matrix_from_edges(dim, edges)
    region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

    if show:
        print('adjacency_matrix:\n', adjacency_matrix)
        print('region_matrix:\n', region_matrix)

    row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)
    matrix_1 = np.zeros((dim, 9, 17))  # max 1-neighbourhood size is 8, plus 1 for center node
    matrix_1[row_indexes, column_indexes, :] = nodes[node_indexes]  # fill in the values

    vector_indexes, node_indexes = utl.get_vector_transformation(adjacency_matrix, region_matrix)
    vector = np.zeros((dim * 9, 17))  # max 1-neighbourhood size is 8, plus 1 for center node
    vector[vector_indexes, :] = nodes[node_indexes]  # fill in the values
    matrix_2 = vector.reshape((dim, 9, 17))

    if show:
        print('matrix_1:\n', matrix_1)
        print('matrix_2:\n', matrix_2)


if __name__ == '__main__':
    main()
    print('end')
