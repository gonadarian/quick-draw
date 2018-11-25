import numpy as np
import models as mdls
import datasets as ds
import utilities as utl
import tensorflow as tf
import keras.backend as k
from keras.callbacks import ModelCheckpoint


preload = True
training = not preload

node_count = 4
edge_count = 4
region_count = 9
encoding_dim = 14

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_nodes_columns(nodes, vector_indexes):
    nodes_columns = k.gather(nodes, vector_indexes)
    assert nodes_columns.shape == (node_count * (region_count + 1), encoding_dim)
    assert nodes_columns.shape == (4 * 9, 17)

    nodes_columns = nodes_columns.reshape((node_count, (region_count + 1) * encoding_dim))
    assert nodes_columns.shape == (4, 9 * 17)

    return nodes_columns


def test_debug():
    batch_size = 3
    channel_size = 14

    with tf.Session() as sess:

        row_idx = tf.reshape(tf.range(batch_size), (-1, 1))  # (?, 1)
        row_idx = tf.tile(row_idx, [1, node_count * region_count])  # (?, 36)
        row_idx = tf.reshape(row_idx, (-1, node_count, region_count))  # (?, 4, 9)
        print(sess.run(row_idx))

        assert row_idx.shape[1:] == (node_count, region_count)  # (?, 4, 9)

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
        assert mapping.shape[1:] == (node_count, region_count)  # (?, 4, 9)

        idx = tf.stack([row_idx, mapping], axis=-1)
        print(sess.run(idx))
        assert idx.shape[1:] == (node_count, region_count, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(idx, empty)
        print(sess.run(where))
        assert where.shape[1:] == (node_count, region_count, 2)

        where = tf.reduce_all(where, axis=3)
        print(sess.run(where))
        assert where.shape[1:] == (node_count, region_count)

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
        assert nodes.shape == (batch_size, node_count, channel_size)  # (?, 4, 14)

        updates = tf.gather_nd(nodes, node_indices)
        print(sess.run(updates))

        nodes_columns = tf.scatter_nd(indices=indices, updates=updates,
                                      shape=(batch_size, node_count, region_count, channel_size))
        print(sess.run(nodes_columns))
        assert nodes_columns.shape[1:] == (node_count, region_count, channel_size)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, node_count, region_count * channel_size))
        print(sess.run(nodes_columns))
        print(sess.run(nodes_columns)[0])
        print(sess.run(nodes_columns)[1][0])
        print('nodes_columns.shape:', nodes_columns.shape[1:])
        assert nodes_columns.shape[1:] == (node_count, region_count * channel_size)  # (?, 4, 126)


def test():
    batch_size = 3
    channel_size = 14

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

    assert mapping.shape[1:] == (node_count, region_count)  # (?, 4, 9)
    assert nodes.shape == (batch_size, node_count, channel_size)  # (?, 4, 14)

    with tf.Session() as sess:

        row_idx = tf.reshape(tf.range(batch_size), (-1, 1))  # (?, 1)
        row_idx = tf.tile(row_idx, [1, node_count * region_count])  # (?, 36)
        row_idx = tf.reshape(row_idx, (-1, node_count, region_count))  # (?, 4, 9)
        assert row_idx.shape[1:] == (node_count, region_count)  # (?, 4, 9)

        idx = tf.stack([row_idx, mapping], axis=-1)
        assert idx.shape[1:] == (node_count, region_count, 2)

        empty = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(idx, empty)
        assert where.shape[1:] == (node_count, region_count, 2)

        where = tf.reduce_all(where, axis=3)
        assert where.shape[1:] == (node_count, region_count)

        indices = tf.where(where)
        assert len(indices.shape) == 2

        node_indices = tf.gather_nd(idx, indices)

        updates = tf.gather_nd(nodes, node_indices)

        nodes_columns = tf.scatter_nd(indices=indices, updates=updates,
                                      shape=(batch_size, node_count, region_count, channel_size))
        assert nodes_columns.shape[1:] == (node_count, region_count, channel_size)  # (?, 4, 9, 14)

        nodes_columns = tf.reshape(nodes_columns, (-1, node_count, region_count * channel_size))
        assert nodes_columns.shape[1:] == (node_count, region_count * channel_size)  # (?, 4, 126)

        print(sess.run(nodes_columns)[0])
        print(sess.run(nodes_columns)[1][0])
        print('nodes_columns.shape:', nodes_columns.shape[1:])


def load_data(full_sample = True):

    if full_sample:
        nodes = ds.load_graph_lines_set()[:, :, 3:]
        mappings = ds.load_graph_mapping_set()

    else:
        nodes = ds.load_graph_lines()
        edges = ds.load_graph_edges()

        regions = utl.get_regions(region_count=8, show=True)

        adjacency_matrix = utl.get_adjacency_matrix_from_edges(node_count, edges)
        region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

        nodes = nodes[:, 3:]  # use 14-dim instead of full 17-dim
        row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

        mappings = np.full((node_count, region_count), -1)
        mappings[row_indexes, column_indexes] = node_indexes

        mappings = mappings.reshape((1, node_count, region_count))
        nodes = nodes.reshape((1, node_count, encoding_dim))

    return nodes, mappings


def main():

    nodes, mappings = load_data()

    if preload:
        autoencoder_model = mdls.load_graph_autoencoder_model(node_count=4, region_count=9)
        autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    else:
        autoencoder_model = mdls.create_graph_autoencoder_model(node_count, encoding_dim, region_count)
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
                # TensorBoard(log_dir='C:\Logs'),
                ModelCheckpoint(
                    'models\graphs\model_autoencoder_v1.{epoch:05d}-{val_loss:.6f}.hdf5',
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
