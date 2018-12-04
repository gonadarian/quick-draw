import numpy as np
import models as mdls
import datasets as ds
import utilities as utl
import tensorflow as tf
import keras.backend as k
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


def test_debug():
    m = 3

    with tf.Session() as sess:

        row_idx = tf.reshape(tf.range(m), (-1, 1))  # (?, 1)
        row_idx = tf.tile(row_idx, [1, vertices * regions])  # (?, 36)
        row_idx = tf.reshape(row_idx, (-1, vertices, regions))  # (?, 4, 9)
        print(sess.run(row_idx))

        assert row_idx.shape[1:] == (vertices, regions)  # (?, 4, 9)

        mapping = tf.constant([[
            [0, -1, 1, -1, -1, 2, -1, -1, 3],
            [0, -1, 1, -1, -1, 2, -1, -1, 3],
            [0, -1, 1, -1, -1, 2, -1, -1, 3],
            [0, -1, 1, -1, -1, 2, -1, -1, 3]], [
            [0, -1, -1, -1, -1, 3, -1, -1, 1],
            [0, -1, -1, -1, -1, 3, -1, -1, 1],
            [0, -1, -1, -1, -1, 3, -1, -1, 1],
            [0, -1, -1, -1, -1, 3, -1, -1, 1]], [
            [0, -1, -1, 2, -1, 1, -1, 3, -1],
            [0, -1, -1, 2, -1, 1, -1, 3, -1],
            [0, -1, -1, 2, -1, 1, -1, 3, -1],
            [0, -1, -1, 2, -1, 1, -1, 3, -1]]])

        print(sess.run(mapping))
        assert mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

        idx = tf.stack([row_idx, mapping], axis=-1)
        print(sess.run(idx))
        assert idx.shape[1:] == (vertices, regions, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(idx, empty)
        print(sess.run(where))
        assert where.shape[1:] == (vertices, regions, 2)

        where = tf.reduce_all(where, axis=3)
        print(sess.run(where))
        assert where.shape[1:] == (vertices, regions)

        indices = tf.where(where)
        print(sess.run(indices))
        assert len(indices.shape) == 2

        node_indices = tf.gather_nd(idx, indices)
        print(sess.run(node_indices))

        nodes = tf.constant([[
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]], [
            [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
            [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
            [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
            [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133]], [
            [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
            [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
            [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
            [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233]]])

        print(sess.run(nodes))
        assert nodes.shape == (m, vertices, channels)  # (?, 4, 14)

        updates = tf.gather_nd(nodes, node_indices)
        print(sess.run(updates))

        nodes_columns = tf.scatter_nd(indices=indices, updates=updates,
                                      shape=(m, vertices, regions, channels))
        print(sess.run(nodes_columns))
        assert nodes_columns.shape[1:] == (vertices, regions, channels)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, vertices, regions * channels))
        print(sess.run(nodes_columns))
        print(sess.run(nodes_columns)[0])
        print(sess.run(nodes_columns)[1][0])
        print('nodes_columns.shape:', nodes_columns.shape[1:])
        assert nodes_columns.shape[1:] == (vertices, regions * channels)  # (?, 4, 126)


def test():
    m = 3

    mapping = tf.constant([[
        [0, -1, 1, -1, -1, 2, -1, -1, 3],
        [0, -1, 1, -1, -1, 2, -1, -1, 3],
        [0, -1, 1, -1, -1, 2, -1, -1, 3],
        [0, -1, 1, -1, -1, 2, -1, -1, 3]], [
        [0, -1, -1, -1, -1, 3, -1, -1, 1],
        [0, -1, -1, -1, -1, 3, -1, -1, 1],
        [0, -1, -1, -1, -1, 3, -1, -1, 1],
        [0, -1, -1, -1, -1, 3, -1, -1, 1]], [
        [0, -1, -1, 2, -1, 1, -1, 3, -1],
        [0, -1, -1, 2, -1, 1, -1, 3, -1],
        [0, -1, -1, 2, -1, 1, -1, 3, -1],
        [0, -1, -1, 2, -1, 1, -1, 3, -1]]])

    nodes = tf.constant([[
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]], [
        [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
        [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
        [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133],
        [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133]], [
        [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
        [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
        [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233],
        [220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233]]])

    assert mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)
    assert nodes.shape == (m, vertices, channels)  # (?, 4, 14)

    with tf.Session() as sess:

        row_idx = tf.reshape(tf.range(m), (-1, 1))  # (?, 1)
        row_idx = tf.tile(row_idx, [1, vertices * regions])  # (?, 36)
        row_idx = tf.reshape(row_idx, (-1, vertices, regions))  # (?, 4, 9)
        assert row_idx.shape[1:] == (vertices, regions)  # (?, 4, 9)

        idx = tf.stack([row_idx, mapping], axis=-1)
        assert idx.shape[1:] == (vertices, regions, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(idx, empty)
        assert where.shape[1:] == (vertices, regions, 2)

        where = tf.reduce_all(where, axis=3)
        assert where.shape[1:] == (vertices, regions)

        indices = tf.where(where)
        assert len(indices.shape) == 2

        node_indices = tf.gather_nd(idx, indices)

        updates = tf.gather_nd(nodes, node_indices)

        nodes_columns = tf.scatter_nd(indices=indices, updates=updates,
                                      shape=(m, vertices, regions, channels))
        assert nodes_columns.shape[1:] == (vertices, regions, channels)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, vertices, regions * channels))
        assert nodes_columns.shape[1:] == (vertices, regions * channels)  # (?, 4, 126)

        print(sess.run(nodes_columns)[0])
        print(sess.run(nodes_columns)[1][0])
        print('nodes_columns.shape:', nodes_columns.shape[1:])


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
