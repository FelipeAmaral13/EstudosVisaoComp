# Bilioteca
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.vis_utils import plot_model

from matplotlib import pyplot
import matplotlib.pyplot as plt


class MNISTClassifier:
    def __init__(self):
        self.num_classes = 10
        self.input_shape = (28, 28, 1)
        self.img_rows, self.img_cols = 28, 28
        self.batch_size = 128
        self.epochs = 10
        self.model = None

    def load_data(self):
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
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1).astype("float32")
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1).astype("float32")

        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test

    def build_model(self):
        # Modelo RNA
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, x_train, y_train, x_test, y_test):
        self.build_model()
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(x_test, y_test))

    def evaluate_model(self, x_test, y_test):
        # Avaliacao simples do modelo
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save_model(self, filename):
        # Salvar o modelo
        self.model.save(filename)

    def plot_model(self):
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        plt.show()


if __name__ == "__main__":
    clf = MNISTClassifier()
    x_train, y_train, x_test, y_test = clf.load_data()
    clf.train_model(x_train, y_train, x_test, y_test)
