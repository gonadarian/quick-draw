import tensorflow as tf


node_count = 4
region_count = 9


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
    row_idx = tf.reshape(row_idx, (-1, node_count, region_count))  # (?, 4, 9)
    assert row_idx.shape[1:] == (node_count, region_count)  # (?, 4, 9)

    idx = tf.stack([row_idx, mapping], axis=-1)  # (?, 4, 9, 2)
    assert idx.shape[1:] == (node_count, region_count, 2)

    empty = tf.constant(-1, dtype=tf.int32)
    where = tf.not_equal(idx, empty)  # (?, 4, 9, 2)
    assert where.shape[1:] == (node_count, region_count, 2)

    where = tf.reduce_all(where, axis=3)  # (?, 4, 9)
    assert where.shape[1:] == (node_count, region_count)

    indices = tf.where(where)  # (?, 3)
    assert indices.shape[1:] == (3, )

    node_indices = tf.gather_nd(idx, indices)  # (?, 2)
    assert node_indices.shape[1:] == (2, )

    updates = tf.gather_nd(nodes, node_indices)  # (?, 14)
    assert updates.shape[1:] == (channel_size, )

    nodes_columns = tf.scatter_nd(indices, updates, shape=(batch_size, node_count, region_count, channel_size))
    assert nodes_columns.shape[1:] == (node_count, region_count, channel_size)  # (?, 4, 9, 14)
    nodes_columns = tf.reshape(nodes_columns, (-1, node_count, region_count * channel_size))
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
