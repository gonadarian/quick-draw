import numpy as np
import random as rand
import utilities as utl
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam

preload = False
train = False
predict = False
cluster = True

rand.seed(1)
np.random.seed(1)


# TODO: move to models
def get_model():
    input_data = Input(shape=(34,))
    x = input_data
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    output_data = x
    model = Model(inputs=input_data, outputs=output_data)
    return model


def get_data():
    X = np.load('generator\data\encoding_clusters_v2_7234x10x2x17.npy')
    assert X.shape == (7234, 10, 2, 17)
    X = X.reshape((72340, 34))
    m = X.shape[0]

    Y = np.tile(np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]), m // 10).reshape((m, 1))
    assert Y.shape == (72340, 1)

    return X, Y


if preload:
    from keras.models import load_model
    model = load_model('models\pairs_encoded\model_dense_v1-088-0.001555.hdf5')

elif train or predict:
    model = get_model()
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()


if train:
    from keras.callbacks import TensorBoard, ModelCheckpoint
    X, Y = get_data()
    model.fit(
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
    X_image = np.load('generator\data\line_samples_v2_7234x28x28x1.npy')
    assert X_image.shape == (7234, 28, 28, 1)
    m = X_image.shape[0]
    indexes = np.random.randint(m, size=2)
    samples = X_image[indexes]

    encoder = load_model('models\lines_encoded\lines_mixed_encoded_v2-091-0.000072.hdf5')
    Y_image_1 = utl.get_embeddings(encoder, samples[0, :, :, 0])
    Y_image_2 = utl.get_embeddings(encoder, samples[1, :, :, 0])
    Y_mix = np.concatenate((Y_image_1, Y_image_2), axis=0)
    m_mix = len(Y_mix)
    assert Y_mix.shape == (m_mix, 17)

    cluster_matrix = utl.calculate_cluster_matrix(model, Y_mix)


if cluster:
    cluster_matrix = np.load('generator\data\cluster_matrix-square-36x36.npy')
    clusters = utl.extract_clusters(cluster_matrix)


print('end')
