import tensorflow as tf


vertices = 4
regions = 9
channels = 14


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


if __name__ == '__main__':
    test()
    test_debug()
    print('end')
