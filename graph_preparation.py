import numpy as np
import utilities as utl

nodes = np.load('generator/data/graph_lines.npy')
edges = np.load('generator/data/graph_edges.npy')
assert nodes.shape == (4, 17)
assert edges.shape == (4, 2)
edge_count = len(edges)

print('nodes:', nodes)
print('edges:', edges)

regions = utl.get_regions(region_count=8, show=True)
region_matrix = utl.get_region_matrix(nodes, regions, show=False)
print(region_matrix[:, :, 4])


print('end')
