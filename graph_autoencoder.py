import time as t
import numpy as np
import keras.backend as k
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


training = False
preload = not training

vertices = 4
regions = 9
channels_full = 17
channels = 14

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_nodes_columns(nodes, vector_indexes):
    nodes_columns = k.gather(nodes, vector_indexes)
    assert nodes_columns.shape == (vertices * regions, channels_full)
    nodes_columns = nodes_columns.reshape((vertices, regions * channels_full))

    return nodes_columns


def main(concept):

    vertices_list, mappings_list, m = concept.dataset_centered()

    units_list = [17, 20, 26, 32, 26, 20, 14]
    autoencoder_model = (concept.model_autoencoder() if preload else
                         concept.model_autoencoder_creator(units_list, vertices, regions))
    autoencoder_model.summary()

    if training:
        epochs = 10000
        batch_size = 32
        timestamp = int(t.time())

        model_name = 'graph-autoencoder-{}-{}'.format(concept.code, timestamp)
        log_dir = 'C:\Logs\{}-b{}'.format(model_name, batch_size)
        filepath = 'models\{}\{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:05d}-{val_loss:.6f}')

        autoencoder_model.fit(
            x=[vertices_list, mappings_list],
            y=vertices_list,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=([vertices_list, mappings_list], vertices_list),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=100)])

    predicted_vertices_list = autoencoder_model.predict(
        x=[vertices_list, mappings_list])

    print('input:\n', vertices_list)
    print('output:\n', predicted_vertices_list)
    print('diff:\n', vertices_list - predicted_vertices_list)


if __name__ == '__main__':
    main(Concept.SQUARE)
    print('end')
