import numpy as np
import datasets as ds
import utilities as utl
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dense


node_count = 4
edge_count = 4
region_count = 9
encoding_dim = 14


def lambda_graph2col_shape(input_shapes):
    [nodes_shape, mapping_shape] = input_shapes

    batch_size = nodes_shape[0]
    node_size = nodes_shape[1]
    node_encoding_size = nodes_shape[2]
    region_size = mapping_shape[2]
    column_size = node_encoding_size * region_size

    return batch_size, node_size, column_size


def lambda_graph2col(inputs):
    [nodes, mapping] = inputs

    batch_size = tf.shape(nodes)[0]
    channel_size = nodes.shape[2]
    print('nodes.shape:', nodes.shape[1:])
    assert nodes.shape[1:] == (node_count, channel_size)  # (?, 4, 14)
    assert mapping.shape[1:] == (node_count, region_count, )  # (?, 4, 9)

    row_idx = tf.reshape(tf.range(batch_size), (-1, 1))  # (?, 1)
    row_idx = tf.tile(row_idx, [1, node_count * region_count])  # (?, 36)
    row_idx = tf.transpose(row_idx)  # (36, ?)
    row_idx = tf.reshape(row_idx, (-1, node_count, region_count))  # (?, 4, 9)
    assert row_idx.shape[1:] == (node_count, region_count)  # (?, 4, 9)

    idx = tf.stack([row_idx, mapping], axis=-1)
    assert idx.shape[1:] == (node_count, region_count, 2)

    nodes_columns = tf.gather_nd(nodes, idx)
    assert nodes_columns.shape[1:] == (node_count, region_count, channel_size)  # (?, 4, 9, 14)
    nodes_columns = tf.reshape(nodes_columns, (-1, node_count, region_count * channel_size))
    print('nodes_columns.shape:', nodes_columns.shape[1:])
    assert nodes_columns.shape[1:] == (node_count, region_count * channel_size)  # (?, 4, 126)

    return nodes_columns


def lambda_mean_shape(nodes_shape):
    assert nodes_shape[1] == node_count  # (?, 4, 14)
    assert len(nodes_shape) == 3  # (?, 4, 14)

    batch_size = nodes_shape[0]
    encoding_size = nodes_shape[2]

    return batch_size, encoding_size


def lambda_mean(nodes):
    assert nodes.shape[1] == node_count  # (?, 4, 14)
    assert len(nodes.shape) == 3  # (?, 4, 14)

    nodes_average = tf.reduce_mean(nodes, 1)
    assert len(nodes_average.shape) == 2  # (?, 14)

    return nodes_average


def lambda_moments_shape(nodes_shape):
    assert nodes_shape[1] == node_count  # (?, 4, 14)
    assert len(nodes_shape) == 3  # (?, 4, 14)

    batch_size = nodes_shape[0]
    encoding_size = nodes_shape[2]

    return [(batch_size, encoding_size), (batch_size, encoding_size)]


def lambda_moments(nodes):
    assert nodes.shape[1] == node_count  # (?, 4, 14)
    assert len(nodes.shape) == 3  # (?, 4, 14)

    mean, variance = tf.nn.moments(nodes, 1)
    assert len(mean.shape) == 2  # (?, 14)
    assert len(variance.shape) == 2  # (?, 14)

    return [mean, variance]


def lambda_repeat_shape(node_shape):
    assert len(node_shape) == 2  # (?, 14)

    batch_size = node_shape[0]
    encoding_size = node_shape[1]

    return batch_size, node_count, encoding_size


def lambda_repeat(node):
    assert len(node.shape) == 2  # (?, 14)
    channel_size = node.shape[1]

    node_repeated = tf.tile(node, [1, node_count])
    node_repeated = tf.reshape(node_repeated, (-1, node_count, channel_size))
    assert node_repeated.shape[1:] == (node_count, channel_size)  # (?, 4, 14)

    return node_repeated


def get_model_autoencoder():
    input_nodes = Input(shape=(node_count, encoding_dim))
    input_mapping = Input(shape=(node_count, region_count), dtype='int32')

    encoding = input_nodes

    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(20, activation='relu')(encoding)
    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(26, activation='relu')(encoding)
    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(32, activation='relu')(encoding)
    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(26, activation='relu')(encoding)
    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(20, activation='relu')(encoding)
    encoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([encoding, input_mapping])
    encoding = Dense(14, activation='tanh')(encoding)

    # encoding = Lambda(lambda_mean, output_shape=lambda_mean_shape)(encoding)
    encoding, variance = Lambda(lambda_moments, output_shape=lambda_moments_shape)(encoding)
    decoding = encoding
    decoding = Lambda(lambda_repeat, output_shape=lambda_repeat_shape)(decoding)

    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(20, activation='relu')(decoding)
    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(26, activation='relu')(decoding)
    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(32, activation='relu')(decoding)
    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(26, activation='relu')(decoding)
    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(20, activation='relu')(decoding)
    decoding = Lambda(lambda_graph2col, output_shape=lambda_graph2col_shape)([decoding, input_mapping])
    decoding = Dense(14, activation='sigmoid')(decoding)

    model = Model([input_nodes, input_mapping], [decoding, variance])
    model.add_loss(variance)
    model.compile(optimizer='adam', loss=['binary_crossentropy', None])

    return model


def get_nodes_columsn(nodes, vector_indexes):
    nodes_columns = K.gather(nodes, vector_indexes)
    assert nodes_columns.shape == (node_count * (region_count + 1), encoding_dim)
    assert nodes_columns.shape == (4 * 9, 17)

    nodes_columns = nodes_columns.reshape((node_count, (region_count + 1) * encoding_dim))
    assert nodes_columns.shape == (4, 9 * 17)

    return nodes_columns


def main():
    model = get_model_autoencoder()
    model.summary()

    nodes = ds.load_graph_lines()
    edges = ds.load_graph_edges()

    regions = utl.get_regions(region_count=8, show=True)
    adjacency_matrix = utl.get_adjacency_matrix_from_edges(node_count, edges)
    region_matrix = utl.get_region_matrix(nodes, regions, show=True, debug=True)

    nodes = nodes[:, 3:]  # use 14-dim instead of full 17-dim
    row_indexes, column_indexes, node_indexes = utl.get_matrix_transformation(adjacency_matrix, region_matrix)

    mapping = np.zeros((node_count, region_count, encoding_dim))
    mapping[row_indexes, column_indexes, :] = nodes[node_indexes]

    mapping = np.zeros((node_count, region_count))
    mapping[row_indexes, column_indexes] = node_indexes

    nodes = nodes.reshape((1, node_count, encoding_dim))
    mapping = mapping.reshape((1, node_count, region_count, encoding_dim))

    model.fit(
        [nodes, mapping], [nodes, None],
        epochs=1,
        batch_size=1,
        shuffle=True,
        validation_data=([nodes, mapping], [nodes, None]),
        # callbacks=[
        #     TensorBoard(log_dir='C:\Logs'),
        #     ModelCheckpoint(
        #         'models\lines\model_autoencoder_v2.{epoch:02d}-{val_loss:.4f}.hdf5',
        #         monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        # ]
    )


if __name__ == '__main__':
    main()
    print('end')
