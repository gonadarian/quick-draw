import time as t
import libs.utilities as utl
from libs.concepts import Concept
from keras.callbacks import TensorBoard, ModelCheckpoint


dim = 27

train = False
preload = not train
predict = True
analyze = False


def analysis(encoder_model, x):
    m = x.shape[0]

    for idx in range(m):
        sample = [x[[idx], ...]]
        activation = encoder_model.predict(sample)
        print('\t'.join(list(map(lambda it: '{0:.3f}'.format(it), activation[0, 0, 0, :]))))


def main(concept, batch_size=64, period=10):

    autoencoder_model, encoder_model, decoder_model = concept.get_model_autoencoder(trained=preload)
    autoencoder_model.summary()

    x, _ = concept.dataset_centered()

    if train:
        epochs = 1000
        timestamp = int(t.time())

        model_name = 'conv-autoencoder-{}-{}'.format(concept.code, timestamp)
        log_dir = 'logs/{}-b{}'.format(model_name, batch_size)
        filepath = 'models/{}/{}-{}.hdf5'.format(concept.code, model_name, 'e{epoch:04d}-{val_loss:.5f}')

        autoencoder_model.fit(
            x, x,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x, x),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(filepath=filepath, save_best_only=True, period=period)
            ]
        )

    if predict:
        utl.show_predictions(autoencoder_model, x, x)

    # show activations of encoded layer with 14 numbers
    if analyze:
        analysis(encoder_model, x)


if __name__ == '__main__':

    main(Concept.LINE)
    main(Concept.ELLIPSE)
    main(Concept.BEZIER)
    main(Concept.STAR)

    print('end')
