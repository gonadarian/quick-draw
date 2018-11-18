import numpy as np
import models as mdls
import random as rand
import utilities as utl
import matplotlib.pyplot as plt
from scipy import spatial
from keras.callbacks import TensorBoard, ModelCheckpoint


preload = True
train = False
predict = True
analysis = True

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def get_distances(encodings):
    count = encodings.shape[0]
    distances = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            distances[i, j] = spatial.distance.cosine(encodings[i, :], encodings[j, :])

    for idx in range(count):
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), distances[idx, :]))), '\n')

    return distances


def load_data():
    x = np.load('generator\data\line_mixed_samples_v2_7234x28x28x1.npy')
    assert x.shape[1:] == (28, 28, 1)
    m = x.shape[0]
    x = np.reshape(x, (m, 28, 28, 1))
    y = np.load('generator\data\line_mixed_encodings_v2_7234x28x28x17.npy')
    assert y.shape[1:] == (28, 28, 17)
    assert y.shape[0] == m
    return x, y, m


X, Y, m = load_data()


if preload:
    encoder_model = mdls.load_encoder_model()

else:
    encoder_model = mdls.create_encoder_model()
    encoder_model.compile(optimizer='adam', loss='mean_squared_error')


if train:
    encoder_model.fit(
        X, Y,
        epochs=60,
        batch_size=32,
        shuffle=True,
        validation_data=(X, Y),
        callbacks=[
            TensorBoard(log_dir='C:\Logs'),
            ModelCheckpoint(
                'models\lines_encoded\lines_mixed_encoded_v2-{epoch:03d}-{val_loss:.6f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    )


if predict:

    predict_single = False
    predict_multiple = True

    decoder_model = mdls.load_decoder_model()

    if predict_single:

        idx = rand.randint(0, X.shape[0])
        print('random index:', idx)
        sample = X[idx, :, :, 0]
        embeddings = utl.get_embeddings(encoder_model, sample, True)

        if analysis:
            offsets = embeddings[:, 1:3]
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 0]))))
            print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), offsets[:, 1]))))
            dist = get_distances(embeddings)
            utl.show_elbow_curve(embeddings)

        utl.decode_clustered_embeddings(decoder_model, embeddings, 1, True)

    if predict_multiple:

        n = 10
        n_clusters = 2
        fig_rows = n_clusters + 1
        fig_cols = n

        plt.figure(figsize=(fig_rows * fig_cols, 4))
        indexes = rand.sample(range(1, X.shape[0]), n)

        for i in range(n):
            sample = X[indexes[i], :, :, 0]
            embeddings = utl.get_embeddings(encoder_model, sample, False)
            images = utl.decode_clustered_embeddings(decoder_model, embeddings, n_clusters, False)

            # display original
            ax = plt.subplot(fig_rows, n, i + 1)
            plt.imshow(sample)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display cluster reconstructions
            for cluster in range(n_clusters):
                ax = plt.subplot(fig_rows, n, i + 1 + (cluster+1) * n)
                img_reconstruct = images[cluster]
                plt.imshow(img_reconstruct)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()


print('end')
