import time as t
import numpy as np
import random as rand
import libs.datasets as ds
import libs.utilities as utl
import matplotlib.pyplot as plt
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27
channels = 14
channels_full = 17

train = False
preload = not train
predict = True
analyze = False

concepts = {Concept.LINE, Concept.ELLIPSE, Concept.BEZIER, Concept.STAR}
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def remove_concept(concept_set, concept):
    concept_set = concept_set.copy()
    concept_set.remove(concept)
    return concept_set


def prediction(concept, matrix_encoder_model, x):
    predict_single = False
    predict_multiple = True

    _, _, decoder_model = concept.model_autoencoder()

    if predict_single:
        prediction_single(decoder_model, matrix_encoder_model, x)

    if predict_multiple:
        prediction_multiple(decoder_model, matrix_encoder_model, x)


def prediction_multiple(decoder_model, matrix_encoder_model, x):
    n = 10
    n_clusters = 2
    fig_rows = n_clusters + 1
    fig_cols = n

    plt.figure(figsize=(fig_rows * fig_cols, 4))
    indexes = rand.sample(range(1, x.shape[0]), n)

    for i in range(n):
        sample = x[indexes[i], :, :, 0]
        embeddings = utl.get_embeddings(matrix_encoder_model, sample, dim=dim, show=False)
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


def prediction_single(decoder_model, matrix_encoder_model, x):
    idx = rand.randint(0, x.shape[0])
    print('random index:', idx)
    sample = x[idx, :, :, 0]
    embeddings = utl.get_embeddings(matrix_encoder_model, sample, dim=dim, show=True)
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
    matrix_encoder_model = concept.get_model_matrix_encoder(trained=preload)
    matrix_encoder_model.summary()

    x, y, m = concept.get_dataset_mixed()

    # add negative samples with zeros for labels
    x_mix, y_mix, m_mix = ds.load_images_mix_mixed(remove_concept(concepts, concept))
    x = np.vstack((x, x_mix))
    y = np.vstack((y, np.zeros_like(y_mix)))

    if train:
        epochs = 1000
        batch_size = 64
        timestamp = int(t.time())

        model_name = 'conv-matrix-encoder-{}-{}'.format(concept.code, timestamp)
        log_dir = 'logs/{}-b{}'.format(model_name, batch_size)
        filepath = 'models/{}/{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:04d}-{val_loss:.6f}')

        matrix_encoder_model.fit(
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
        prediction(concept, matrix_encoder_model, x)


if __name__ == '__main__':

    main(Concept.LINE)
    main(Concept.ELLIPSE)
    main(Concept.BEZIER)
    main(Concept.STAR)

    print('end')
