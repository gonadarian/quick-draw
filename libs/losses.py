import keras.backend as k


def absolute_loss(vector):
    loss = k.mean(vector)
    assert loss.shape == ()
    return loss
