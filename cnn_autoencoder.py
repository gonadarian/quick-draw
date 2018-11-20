import numpy as np
import models as mdls
import random as rand
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint


preload = True
train = False
predict = True
analyze_1 = True
analyze_2 = True


if preload:
    autoencoder_model = mdls.load_autoencoder_model()

else:
    autoencoder_model = mdls.get_model_autoencoder()
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')


print(autoencoder_model.summary())

X = np.load('generator\data\line_originals_v2_392x28x28.npy')  # TODO use datasets load method
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), 28, 28, 1))
sample_count = X.shape[0]


if train:
    autoencoder_model.fit(
        X, X,
        epochs=400,
        batch_size=64,
        shuffle=True,
        validation_data=(X, X),
        callbacks=[
            TensorBoard(log_dir='C:\Logs'),
            ModelCheckpoint(
                'models\lines\model_autoencoder_v2.{epoch:02d}-{val_loss:.4f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    )


if predict:

    n = 10
    decoded_images = autoencoder_model.predict(x_test)
    indexes = rand.sample(range(1, decoded_images.shape[0]), n)

    plt.figure(figsize=(30, 4))
    for i in range(n):
        img_idx = indexes[i]

        # display original
        ax = plt.subplot(3, n, i + 1)
        img_original = x_test[img_idx].reshape(28, 28)
        plt.imshow(img_original)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        image_reconstruct = decoded_images[img_idx].reshape(28, 28)
        plt.imshow(image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(img_original - image_reconstruct)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# show activations of encoded layer with 14 numbers
if analyze_1:

    autoencoder_model.outputs = [autoencoder_model.layers[8].output]

    for idx in range(sample_count):
        sample = [x_test[[idx], ...]]
        activation = autoencoder_model.predict(sample)
        print('\t'.join(list(map(lambda x: '{0:.3f}'.format(x), activation[0,0,0,:]))))


# generate images for encoded values of choice
if analyze_2:

    decoder_model = mdls.gen_decoder_model(autoencoder_model, show=True)

    # this is a 14-number encoding for one of the lines in the test set
    sample = np.array([[[[-0.266, 0.209, 0.830, -0.031, 0.069, 0.922, -0.987, 0.800, -0.882, 0.431, 0.853, 0.117, 0.793, 0.388]]]])

    # show 10 images for different encoding variations
    for idx in range(10):
        img = decoder_model.predict(sample)
        img = img.reshape(28, 28)

        plt.gray()
        plt.imshow(img)
        plt.show()

        # test what happens when 3rd number is increased
        sample[0, 0, 0, 0] += 0.1


print('end')
