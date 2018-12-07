import time as t
import numpy as np
import random as rand
import libs.models as mdls
import libs.utilities as utl
import matplotlib.pyplot as plt
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27
channels = 14
channels_full = 17

preload = False
train = not preload
predict = True
analyze = False

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def prediction(encoder_model, x):
    predict_single = False
    predict_multiple = True

    decoder_model = mdls.load_decoder_line_model_27x27()

    if predict_single:
        prediction_single(decoder_model, encoder_model, x)

    if predict_multiple:
        prediction_multiple(decoder_model, encoder_model, x)


def prediction_multiple(decoder_model, encoder_model, x):
    n = 10
    n_clusters = 2
    fig_rows = n_clusters + 1
    fig_cols = n

    plt.figure(figsize=(fig_rows * fig_cols, 4))
    indexes = rand.sample(range(1, x.shape[0]), n)

    for i in range(n):
        sample = x[indexes[i], :, :, 0]
        embeddings = utl.get_embeddings(encoder_model, sample, dim=dim, show=False)
        images = utl.decode_clustered_embeddings(decoder_model, embeddings, n_clusters, dim=dim, show=False)

        # display original
        ax = plt.subplot(fig_rows, n, i + 1)
        plt.imshow(sample)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display cluster reconstructions
        for cluster in range(n_clusters):
            ax = plt.subplot(fig_rows, n, i + 1 + (cluster + 1) * n)
            img_reconstruct = images[cluster]
            plt.imshow(img_reconstruct)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def prediction_single(decoder_model, encoder_model, x):
    idx = rand.randint(0, x.shape[0])
    print('random index:', idx)
    sample = x[idx, :, :, 0]
    embeddings = utl.get_embeddings(encoder_model, sample, dim=dim, show=True)
    utl.decode_clustered_embeddings(decoder_model, embeddings, 1, True)

    if analyze:
        analysis(embeddings)


def analysis(embeddings):
    offsets = embeddings[:, 1:3]
    print('offsets:')
    print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), offsets[:, 0]))))
    print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), offsets[:, 1]))))
    utl.show_distances(embeddings)
    utl.show_elbow_curve(embeddings)


def main(concept):
    x, y, m = concept.dataset_mixed()

    if preload:
        encoder_model = concept.model_matrix_encoder()

    else:
        encoder_model = concept.model_matrix_encoder_creator()
        encoder_model.compile(optimizer='adam', loss='mean_squared_error')

    if train:
        epochs = 1000
        batch_size = 64
        timestamp = int(t.time())

        model_name = 'conv-matrix-encoder-{}-{}'.format(concept.code, timestamp)
        log_dir = 'C:\Logs\{}-b{}'.format(model_name, batch_size)
        filepath = 'models\{}\{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:04d}-{val_loss:.6f}')

        encoder_model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x, y),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=1)
            ])

    if predict:
        prediction(encoder_model, x)


if __name__ == '__main__':
    main(Concept.LINE)
    main(Concept.ELLIPSE)
    print('end')
