#! python3

# Bilioteca
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model

from matplotlib import pyplot
import matplotlib.pyplot as plt

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Data, split
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# plotar 9 primeiras imagens
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
plt.show()


# resizing e normalizacao da imagem
img_rows, img_cols = 28, 28
x_train = x_train.reshape(
    x_train.shape[0], img_rows, img_cols, 1).astype("float32")
x_test = x_test.reshape(
    x_test.shape[0], img_rows, img_cols, 1).astype("float32")

x_train /= 255
x_test /= 255


# Modelo RNA
model = Sequential()
model.add(
    Conv2D(
        32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(img_rows, img_cols, 1)
        ))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.summary()

plot_model(
    model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plt.show()

# Treinamento
batch_size = 128
epochs = 10

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Avaliacao simples do modelo
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Salvar o modelo
model.save("test_model.h5")
