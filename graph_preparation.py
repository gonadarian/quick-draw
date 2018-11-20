import numpy as np
import utilities as utl
import datasets as ds


dim = 4
real_data = True


nodes = ds.load_graph_lines() if real_data else np.arange(dim * 17).reshape((dim, 17))
assert nodes.shape == (dim, 17)

edges = ds.load_graph_edges()
assert edges.shape == (dim, 2)

print('nodes:', nodes)
print('edges:', edges)

regions = utl.get_regions(region_count=8, show=True)
region_matrix = utl.get_region_matrix(nodes, regions, show=False)
print(region_matrix)

matrix = np.zeros((dim, 9, 17))  # max 1-neighbourhood size is 8, plus 1 for center node
adjacency_matrix = utl.get_adjacency_matrix_from_edges(dim, edges)
row_indexes, column_indexes, cell_vectors = utl.get_matrix_transformation(adjacency_matrix, region_matrix, nodes)

matrix[row_indexes, column_indexes, :] = cell_vectors  # fill in the values


print('end')
