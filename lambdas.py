import tensorflow as tf


# TODO move to libs


def lambda_graph2col_shape(input_shapes):
    [nodes_shape, mapping_shape] = input_shapes

    m = nodes_shape[0]
    vertices = nodes_shape[1]
    channels = nodes_shape[2]
    regions = mapping_shape[2]
    columns = channels * regions

    return m, vertices, columns


def lambda_graph2col(inputs):
    [nodes, mapping] = inputs

    m = tf.shape(nodes)[0]
    vertices = nodes.shape[1]
    channels = nodes.shape[2]
    regions = mapping.shape[2]
    assert nodes.shape[1:] == (vertices, channels)  # (?, 4, 14)
    assert mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

    row_mapping = tf.reshape(tf.range(m), (-1, 1))  # (?, 1)
    row_mapping = tf.tile(row_mapping, [1, vertices * regions])  # (?, 36)
    row_mapping = tf.reshape(row_mapping, (-1, vertices, regions))  # (?, 4, 9)
    assert row_mapping.shape[1:] == (vertices, regions)  # (?, 4, 9)

    nodes_mapping = tf.stack([row_mapping, mapping], axis=-1)  # (?, 4, 9, 2)
    assert nodes_mapping.shape[1:] == (vertices, regions, 2)

    empty = tf.constant(-1, dtype=tf.int32)
    non_empty_mask = tf.not_equal(nodes_mapping, empty)  # (?, 4, 9, 2)
    assert non_empty_mask.shape[1:] == (vertices, regions, 2)
    non_empty_mask = tf.reduce_all(non_empty_mask, axis=3)  # (?, 4, 9)
    assert non_empty_mask.shape[1:] == (vertices, regions)
    non_empty_indices = tf.where(non_empty_mask)  # (?, 3)
    assert non_empty_indices.shape[1:] == (3, )

    mapped_nodes_indices = tf.gather_nd(nodes_mapping, non_empty_indices)  # (?, 2)
    assert mapped_nodes_indices.shape[1:] == (2, )

    mapped_nodes = tf.gather_nd(nodes, mapped_nodes_indices)  # (?, 14)
    assert mapped_nodes.shape[1:] == (channels, )

    nodes_columns = tf.scatter_nd(
        indices=non_empty_indices,
        updates=mapped_nodes,
        shape=(m, vertices, regions, channels))
    assert nodes_columns.shape[1:] == (vertices, regions, channels)  # (?, 4, 9, 14)

    nodes_columns = tf.reshape(nodes_columns, (-1, vertices, regions * channels))
    assert nodes_columns.shape[1:] == (vertices, regions * channels)  # (?, 4, 126)

    return nodes_columns


def lambda_mean_shape(nodes_shape):
    assert len(nodes_shape) == 3  # (?, 4, 14)

    m = nodes_shape[0]
    channels = nodes_shape[2]

    return m, channels


def lambda_mean(nodes):
    assert len(nodes.shape) == 3  # (?, 4, 14)

    nodes_average = tf.reduce_mean(nodes, axis=1)
    assert len(nodes_average.shape) == 2  # (?, 14)

    return nodes_average


def lambda_moments_shape(nodes_shape):
    assert len(nodes_shape) == 3  # (?, 4, 14)

    m = nodes_shape[0]
    channels = nodes_shape[2]

    return [(m, channels), (m, channels)]


def lambda_moments(nodes):
    assert len(nodes.shape) == 3  # (?, 4, 14)

    mean, variance = tf.nn.moments(nodes, axes=1)
    assert len(mean.shape) == 2  # (?, 14)
    assert len(variance.shape) == 2  # (?, 14)

    return [mean, variance]


def lambda_repeat_shape(node_shape):
    assert len(node_shape) == 2  # (?, 14)

    vertices = 4  # TODO can not remain hardcoded
    m = node_shape[0]
    channels = node_shape[1]

    return m, vertices, channels


def lambda_repeat(node):
    assert len(node.shape) == 2  # (?, 14)

    vertices = 4  # TODO can not remain hardcoded
    channels = node.shape[1]

    node_repeated = tf.tile(node, [1, vertices])
    node_repeated = tf.reshape(node_repeated, (-1, vertices, channels))
    assert node_repeated.shape[1:] == (vertices, channels)  # (?, 4, 14)

    return node_repeated
