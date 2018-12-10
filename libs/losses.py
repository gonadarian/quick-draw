import numpy as np
import keras.backend as k
import tensorflow as tf


def absolute_loss(vector):
    loss = k.mean(vector)
    assert loss.shape == ()
    return loss


def masked_mean_squared_error(masked_value=-1):

    def _masked_mean_squared_error(y_true, y_pred):
        # find all masked values and form boolean matrix
        mask = y_true == masked_value

        # make prediction have equal value in masked locations, so it would get ignored by the difference sum
        y_pred[mask] = masked_value

        # count how many non mapped values are there, so that mean can be calculated from the sum
        unmasked_count = tf.cast(tf.count_nonzero(tf.logical_not(mask), axis=-1), dtype=y_true.dtype)

        # calculate the mean, by ignoring masked values
        loss = tf.reduce_sum(k.square(y_pred - y_true), axis=-1) / unmasked_count

        return loss

    return _masked_mean_squared_error


def _test():

    def mean_squared_error(val_true, val_pred):
        return k.mean(k.square(val_pred - val_true), axis=-1)

    def calc_loss(val_true, val_pred, loss_fn):
        loss = loss_fn(val_true, val_pred)
        loss = k.eval(loss)
        print('\ty_true:\n', val_true)
        print('\ty_pred:\n', val_pred)
        print('\tloss:\n', loss)
        print('\tmask count:\n', k.eval(tf.count_nonzero(val_true == -1)))

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    y_true = np.ones((4, ))
    y_pred = np.array([2.0, 3.0, 4.0, 5.0])
    print('baseline')
    calc_loss(y_true, y_pred, masked_mean_squared_error(-1))

    y_true[1] = -1

    # print('masked old')
    # calc_loss(y_true, y_pred, mean_squared_error)
    print('masked new')
    calc_loss(y_true, y_pred, masked_mean_squared_error(-1))
    print('masked old cropped')
    mask = y_true != -1
    calc_loss(y_true[mask], y_pred[mask], mean_squared_error)

    y_true = np.ones((2, 3))
    y_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print('baseline')
    calc_loss(y_true, y_pred, masked_mean_squared_error(-1))

    y_true[0, 1] = -1

    # print('masked old')
    # calc_loss(y_true, y_pred, mean_squared_error)
    print('masked new')
    calc_loss(y_true, y_pred, masked_mean_squared_error(-1))


if __name__ == '__main__':
    _test()
