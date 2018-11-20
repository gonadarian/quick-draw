import numpy as np
import models as mdls
import random as rand
import utilities as utl
from keras.callbacks import TensorBoard, ModelCheckpoint


preload = True
train = False
predict = True
cluster = True

rand.seed(1)
np.random.seed(1)


def get_data():
    X = np.load('generator\data\encoding_clusters_v2_7234x10x2x17.npy')  # TODO use datasets load method
    assert X.shape == (7234, 10, 2, 17)

    X = X.reshape((72340, 34))
    m = X.shape[0]

    Y = np.tile(np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]), m // 10).reshape((m, 1))
    assert Y.shape == (72340, 1)

    return X, Y


if preload:
    clustering_model = mdls.load_clustering_model()

elif train or predict:
    clustering_model = mdls.create_clustering_model()
    clustering_model.compile(optimizer='adam', loss='binary_crossentropy')


if train:
    X, Y = get_data()
    clustering_model.fit(
        X, Y,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_data=(X, Y),
        callbacks=[
            TensorBoard(log_dir='C:\Logs'),
            ModelCheckpoint(
                'models\pairs_encoded\model_dense_v1-{epoch:03d}-{val_loss:.6f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    )


if predict:
    X_image = np.load('generator\data\line_samples_v2_7234x28x28x1.npy')  # TODO use datasets load method
    assert X_image.shape == (7234, 28, 28, 1)

    m = X_image.shape[0]
    indexes = np.random.randint(m, size=2)
    samples = X_image[indexes]

    encoder_model = mdls.load_encoder_model()

    Y_image_1 = utl.get_embeddings(encoder_model, samples[0, :, :, 0])
    Y_image_2 = utl.get_embeddings(encoder_model, samples[1, :, :, 0])
    Y_mix = np.concatenate((Y_image_1, Y_image_2), axis=0)
    m_mix = len(Y_mix)
    assert Y_mix.shape == (m_mix, 17)

    cluster_matrix = utl.calculate_cluster_matrix(clustering_model, Y_mix)


if cluster:
    cluster_matrix = np.load('generator\data\cluster_matrix-square-36x36.npy')  # TODO use datasets load method


if predict or cluster:
    clusters = utl.extract_clusters(cluster_matrix)
    print(clusters)


print('end')
