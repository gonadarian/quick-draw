import time as t
import numpy as np
import random as rand
import libs.utilities as utl
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

train = True
preload = not train
predict = True

# TODO extract to utility method and reuse where needed
rand.seed(1)
np.random.seed(1)


def prediction(concept, clustering_model):
    x, _, m = concept.dataset_shifted()
    indices = np.random.randint(m, size=2)
    x_pair = x[indices]

    matrix_encoder_model = concept.model_matrix_encoder()
    sample_threshold = concept.sample_threshold

    embeddings_1 = utl.get_embeddings(matrix_encoder_model, x_pair[0, :, :, 0], dim=dim, threshold=sample_threshold)
    embeddings_2 = utl.get_embeddings(matrix_encoder_model, x_pair[1, :, :, 0], dim=dim, threshold=sample_threshold)
    embeddings_mix = np.concatenate((embeddings_1, embeddings_2), axis=0)
    assert embeddings_mix.shape[1:] == (17,)

    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, embeddings_mix)
    clusters = utl.extract_clusters(cluster_matrix)
    print(clusters)


def main(concept, batch_size=32):

    clustering_model = concept.get_model_clustering(trained=preload)
    clustering_model.summary()

    x, y, _ = concept.dataset_clustered()

    if train:

        epochs = 1000
        timestamp = int(t.time())

        model_name = 'conv-clustering-{}-{}'.format(concept.code, timestamp)
        log_dir = 'logs/{}-b{}'.format(model_name, batch_size)
        filepath = 'models/{}/{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:04d}-{val_loss:.6f}')

        clustering_model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x, y),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=1)
            ]
        )

    if predict:
        prediction(concept, clustering_model)


if __name__ == '__main__':

    main(Concept.LINE)
    main(Concept.ELLIPSE)
    main(Concept.BEZIER)
    main(Concept.STAR, batch_size=512)

    print('end')
