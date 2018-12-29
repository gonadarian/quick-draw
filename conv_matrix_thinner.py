import time as t
import numpy as np
import libs.models as mdls
import libs.datasets as ds
import libs.utilities as utl
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

train = False
preload = True
predict = True

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def main():

    if preload:
        matrix_thinner_model = mdls.load_matrix_thinner_mix_model()
    else:
        matrix_thinner_model = mdls.create_matrix_thinner_model()
        matrix_thinner_model.summary()

    x, y, m = ds.load_images_mix_thinned()
    print('x:', x.shape)
    print('y:', y.shape)

    if train:
        epochs = 100
        initial_epoch = 0
        batch_size = 64
        timestamp = int(t.time())

        model_name = 'conv-matrix-thinner-mix-{}'.format(timestamp)
        log_dir = 'C:\Logs\{}-b{}'.format(model_name, batch_size)
        filepath = 'models\mix\{}-{}.hdf5'.format(model_name, 'e{epoch:04d}-{val_loss:.6f}')

        matrix_thinner_model.fit(
            x, y,
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x, y),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=1)
            ])

    if predict:
        utl.show_predictions(matrix_thinner_model, x, y)


if __name__ == '__main__':
    main()
    print('end')
