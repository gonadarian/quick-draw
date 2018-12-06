import time as t
import numpy as np
import random as rand
import libs.models as mdls
import libs.utilities as utl
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27
preload = False
train = not preload
predict = False
cluster = False

rand.seed(1)
np.random.seed(1)


def get_data():
    # TODO use datasets load method
    x = np.load('generator\data\lines_27x27\line_27x27_clusters_v1_5815x10x2x17.npy')
    assert x.shape == (5815, 10, 2, 17)

    x = x.reshape((58150, 34))
    m = x.shape[0]

    y = np.tile(np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]), m // 10).reshape((m, 1))
    assert y.shape == (58150, 1)

    return x, y


def main():
    clustering_model = None

    if preload:
        clustering_model = mdls.load_clustering_model()

    elif train or predict:
        clustering_model = mdls.create_clustering_model()
        clustering_model.compile(optimizer='adam', loss='binary_crossentropy')

    if train:
        x, y = get_data()
        clustering_model.fit(
            x, y,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_data=(x, y),
            callbacks=[
                TensorBoard(log_dir='C:\Logs\Dense Clustering v1.b32.{}'.format(int(t.time()))),
                ModelCheckpoint(
                    'models\lines_27x27\model_clustering_v1-{epoch:03d}-{val_loss:.6f}.hdf5',
                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            ]
        )

    cluster_matrix = None

    if predict:
        # TODO use datasets load method
        x_image = np.load('generator\data\lines_27x27\line_27x27_samples_v1_5815x27x27x1.npy')
        assert x_image.shape == (5815, dim, dim, 1)
        m = x_image.shape[0]

        indexes = np.random.randint(m, size=2)
        samples = x_image[indexes]

        encoder_model = mdls.load_encoder_model()

        y_image_1 = utl.get_embeddings(encoder_model, samples[0, :, :, 0])
        y_image_2 = utl.get_embeddings(encoder_model, samples[1, :, :, 0])
        y_mix = np.concatenate((y_image_1, y_image_2), axis=0)
        m_mix = len(y_mix)
        assert y_mix.shape == (m_mix, 17)

        cluster_matrix = utl.calculate_cluster_matrix(clustering_model, y_mix)

    if cluster:
        # TODO use datasets load method
        cluster_matrix = np.load('generator\data\cluster_matrix-square-36x36.npy')

    if predict or cluster:
        clusters = utl.extract_clusters(cluster_matrix)
        print(clusters)


if __name__ == '__main__':
    main()
    print('end')
