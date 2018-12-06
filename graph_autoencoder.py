import numpy as np
import keras.backend as k
import libs.models as mdls
import libs.datasets as ds
import libs.utilities as utl
from keras.callbacks import TensorBoard, ModelCheckpoint


preload = True
training = not preload

vertices = 4
edge_count = 4
regions = 9
channels_full = 17
channels = 14

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_nodes_columns(nodes, vector_indexes):
    nodes_columns = k.gather(nodes, vector_indexes)
    assert nodes_columns.shape == (vertices * regions, channels_full)
    nodes_columns = nodes_columns.reshape((vertices, regions * channels_full))

    return nodes_columns


def load_data(full_sample=True):

    if full_sample:
        node_list = ds.load_graph_lines_set()
        mapping_list = ds.load_graph_mapping_set()

    else:
        node_list = ds.load_graph_lines()
        edge_list = ds.load_graph_edges()

        adjacency_matrix = utl.get_adjacency_matrix_from_edges(vertices, edge_list)
        region_list = utl.get_regions(regions)
        region_matrix = utl.get_region_matrix(node_list, region_list, show=True, debug=True)

        row_indices, column_indices, node_indices = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

        mapping_list = np.full((vertices, regions), -1)
        mapping_list[row_indices, column_indices] = node_indices

        mapping_list = mapping_list.reshape((1, vertices, regions))
        node_list = node_list.reshape((1, vertices, channels_full))

    return node_list, mapping_list


def main():

    nodes, mappings = load_data()

    if preload:
        autoencoder_model = mdls.load_graph_autoencoder_model(vertices, regions, version=4)

    else:
        autoencoder_model = mdls.create_graph_autoencoder_model(
            units_list=[17, 20, 26, 32, 26, 20, 14],
            node_count=vertices,
            region_count=regions)

    autoencoder_model.summary()

    if training:
        autoencoder_model.fit(
            x=[nodes, mappings],
            y=nodes,
            batch_size=32,
            epochs=10000,
            shuffle=True,
            validation_data=([nodes, mappings], nodes),
            callbacks=[
                TensorBoard(log_dir='C:\Logs\Graph Autoencoder v5.b32'),
                ModelCheckpoint(
                    'models\graphs\model_autoencoder_v5.b32.{epoch:05d}-{val_loss:.6f}.hdf5',
                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                    mode='auto', period=100)
            ]
        )

    decoded_nodes = autoencoder_model.predict(x=[nodes, mappings])

    print('input:\n', nodes)
    print('output:\n', decoded_nodes)
    print('diff:\n', nodes - decoded_nodes)


if __name__ == '__main__':
    main()
    print('end')
