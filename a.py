from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
# from Make_layers import MyModel
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.activations import linear, softmax
from tensorflow_addons.activations import lisht
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.activations import linear, softmax
from tensorflow_addons.activations import lisht
import tensorflow.keras.layers

tf.compat.v1.disable_v2_behavior()


class MyModel(tensorflow.keras.Model):
    tf.keras.backend.set_floatx('float64')

    def __init__(self, input_col1_shape, input_col2_shape, input_col3_shape, input_col4_shape,
                 input_col5_shape, num_col, dynamic):
        super(MyModel, self).__init__()
        self.run_eagerly = True

        self.input_col1_shape = input_col1_shape
        self.input_col2_shape = input_col2_shape
        self.input_col3_shape = input_col3_shape
        self.input_col4_shape = input_col4_shape
        self.input_col5_shape = input_col5_shape

        self.num_col = num_col

        # First column.

        self.col1_conv2D1 = Conv2D(filters=20, kernel_size=(4, 4), input_shape=input_col1_shape,
                               padding='valid', activation=lisht)

        self.col1_maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.col1_maxPool1_act = Activation(linear)

        self.col1_conv2D2 = Conv2D(filters=40, kernel_size=5,
                               padding='valid', activation=lisht)

        self.col1_maxPool2 = MaxPooling2D(pool_size=(3, 3))
        self.col1_maxPool2_act = Activation(linear)

        self.col1_flatten = Flatten()

        self.col1_dense1 = Dense(units=150, activation=lisht)

        self.col1_dense2 = Dense(units=10, activation=softmax)

        # Second column.
        self.col2_conv2D1 = Conv2D(filters=20, kernel_size=(4, 4), input_shape=input_col2_shape,
                                   padding='valid', activation=lisht)

        self.col2_maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.col2_maxPool1_act = Activation(linear)

        self.col2_conv2D2 = Conv2D(filters=40, kernel_size=5,
                                   padding='valid', activation=lisht)

        self.col2_maxPool2 = MaxPooling2D(pool_size=(3, 3))
        self.col2_maxPool2_act = Activation(linear)

        self.col2_flatten = Flatten()

        self.col2_dense1 = Dense(units=150, activation=lisht)

        self.col2_dense2 = Dense(units=10, activation=softmax)

        # Third column.
        self.col3_conv2D1 = Conv2D(filters=20, kernel_size=(4, 4), input_shape=input_col3_shape,
                                   padding='valid', activation=lisht)

        self.col3_maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.col3_maxPool1_act = Activation(linear)

        self.col3_conv2D2 = Conv2D(filters=40, kernel_size=5,
                                   padding='valid', activation=lisht)

        self.col3_maxPool2 = MaxPooling2D(pool_size=(3, 3))
        self.col3_maxPool2_act = Activation(linear)

        self.col3_flatten = Flatten()

        self.col3_dense1 = Dense(units=150, activation=lisht)

        self.col3_dense2 = Dense(units=10, activation=softmax)

        # Fourth column.
        self.col4_conv2D1 = Conv2D(filters=20, kernel_size=(4, 4), input_shape=input_col4_shape,
                                   padding='valid', activation=lisht)

        self.col4_maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.col4_maxPool1_act = Activation(linear)

        self.col4_conv2D2 = Conv2D(filters=40, kernel_size=5,
                                   padding='valid', activation=lisht)

        self.col4_maxPool2 = MaxPooling2D(pool_size=(3, 3))
        self.col4_maxPool2_act = Activation(linear)

        self.col4_flatten = Flatten()

        self.col4_dense1 = Dense(units=150, activation=lisht)

        self.col4_dense2 = Dense(units=10, activation=softmax)

        # Fifth column.
        self.col5_conv2D1 = Conv2D(filters=20, kernel_size=(4, 4), input_shape=input_col5_shape,
                                   padding='valid', activation=lisht)

        self.col5_maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.col5_maxPool1_act = Activation(linear)

        self.col5_conv2D2 = Conv2D(filters=40, kernel_size=5,
                                   padding='valid', activation=lisht)

        self.col5_maxPool2 = MaxPooling2D(pool_size=(3, 3))
        self.col5_maxPool2_act = Activation(linear)

        self.col5_flatten = Flatten()

        self.col5_dense1 = Dense(units=150, activation=lisht)

        self.col5_dense2 = Dense(units=10, activation=softmax)


    # def build(self, input_shape):


    def call(self, inputs):
        inputs_col1 = tf.image.resize(inputs, size=self.input_col1_shape[0:2])
        output_col1 = self.col1_conv2D1(inputs_col1)
        output_col1 = self.col1_maxPool1_act(self.col1_maxPool1(output_col1))
        output_col1 = self.col1_conv2D2(output_col1)
        output_col1 = self.col1_maxPool2_act(self.col1_maxPool2(output_col1))
        output_col1 = self.col1_flatten(output_col1)
        output_col1 = self.col1_dense1(output_col1)
        output_col1 = self.col1_dense2(output_col1)

        inputs_col2 = tf.image.resize(inputs, size=self.input_col2_shape[0:2])
        output_col2 = self.col2_conv2D1(inputs_col2)
        output_col2 = self.col2_maxPool1_act(self.col2_maxPool1(output_col2))
        output_col2 = self.col2_conv2D2(output_col2)
        output_col2 = self.col2_maxPool2_act(self.col2_maxPool2(output_col2))
        output_col2 = self.col2_flatten(output_col2)
        output_col2 = self.col2_dense1(output_col2)
        output_col2 = self.col2_dense2(output_col2)

        inputs_col3 = tf.image.resize(inputs, size=self.input_col3_shape[0:2])
        output_col3 = self.col3_conv2D1(inputs_col3)
        output_col3 = self.col3_maxPool1_act(self.col3_maxPool1(output_col3))
        output_col3 = self.col3_conv2D2(output_col3)
        output_col3 = self.col3_maxPool2_act(self.col3_maxPool2(output_col3))
        output_col3 = self.col3_flatten(output_col3)
        output_col3 = self.col3_dense1(output_col3)
        output_col3 = self.col3_dense2(output_col3)

        inputs_col4 = tf.image.resize(inputs, size=self.input_col4_shape[0:2])
        output_col4 = self.col4_conv2D1(inputs_col4)
        output_col4 = self.col4_maxPool1_act(self.col4_maxPool1(output_col4))
        output_col4 = self.col4_conv2D2(output_col4)
        output_col4 = self.col4_maxPool2_act(self.col4_maxPool2(output_col4))
        output_col4 = self.col4_flatten(output_col4)
        output_col4 = self.col4_dense1(output_col4)
        output_col4 = self.col4_dense2(output_col4)

        inputs_col5 = tf.image.resize(inputs, size=self.input_col5_shape[0:2])
        output_col5 = self.col5_conv2D1(inputs_col5)
        output_col5 = self.col5_maxPool1_act(self.col5_maxPool1(output_col5))
        output_col5 = self.col5_conv2D2(output_col5)
        output_col5 = self.col5_maxPool2_act(self.col5_maxPool2(output_col5))
        output_col5 = self.col5_flatten(output_col5)
        output_col5 = self.col5_dense1(output_col5)
        output_col5 = self.col5_dense2(output_col5)

        output_stack = tf.stack([output_col1, output_col2, output_col3, output_col4, output_col5])

        # proto_tensor = tf.make_tensor_proto(output_col3)
        # print((output_col3 + output_col1))
        # return result
        g = tf.compat.v1.get_default_graph()
        with g.as_default():
            a = output_col1 + output_col3
            return g.run(a)



def read_train_data(n, path):
    train_X = np.zeros((10 * n, 28, 28))
    train_y = np.zeros((10 * n, 1))

    for i in range(10):
        print(f'Reading training data from class {i + 1}')

        # Reading image in each class.
        for j in range(n):
            train_X[i * n + j, :, :] = image.imread(f"{path}/{i + 1}/" + "{:05d}.jpg".format(j + 1))
            train_y[i * n + j] = i

    train_X = np.expand_dims(train_X, axis=-1)

    return train_X, train_y


def read_train_data_as_text(n, path):
    train_X = np.zeros((10 * n, 28, 28))
    train_y = np.zeros((10 * n, 1))

    for i in range(10):
        print(f'Reading training data from class {i + 1}')

        # Reading image in each class.
        for j in range(n):
            train_X[i * n + j, :, :] = image.imread(f"{path}/{i + 1}/0" + f"{5941 + j}.jpg")
            train_y[i * n + j] = i

    train_X = np.expand_dims(train_X, axis=-1)

    return train_X, train_y


def read_test_data(m, path):
    test_X = np.zeros((m, 28, 28))

    for i in range(m):
        test_X[i, :, :] = image.imread(f"{path}/test/" + "{:05d}.jpg".format(i))

    test_X = np.expand_dims(test_X, axis=-1)
    return test_X

def normalize(trdata, X):
    trdata_norm = trdata.astype('float32')
    X_norm = X.astype('float32')
    trdata_norm = trdata_norm/255.0
    X_norm = X_norm/255.0
    return trdata_norm, X_norm


def main():
    path_train = "tau-ethiopic-digit-recognition/train"
    path_test = "tau-ethiopic-digit-recognition/test"

    train_X, train_y = read_train_data(100, path_train)
    test_X, test_y = read_train_data_as_text(60, path_test)

    # Flatten the images
    # train_X, test_X = normalize(train_X, test_X)
    # train_X = train_X.reshape((-1, 28*28))
    # test_X = test_X.reshape((-1, 28*28))
    print(train_X.shape)
    print(test_X.shape)
    print(train_y.shape)

    N = train_X.shape[0]
    batch_size = 100
    step_in_epoch = N // batch_size
    num_of_epoch = 50

    model = MyModel(input_col1_shape=(28, 28, 1), input_col2_shape=(28, 28, 1),
                    input_col3_shape=(26, 26, 1), input_col4_shape=(22, 22, 1),
                    input_col5_shape=(20, 20, 1), num_col=5, dynamic=True)

    # a = 0.00003
    init_learning_rate = 0.001
    lr_schedule = ExponentialDecay(init_learning_rate, decay_steps=step_in_epoch, decay_rate=0.993)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=["acc"])

    X = np.array([train_X[0, :, :, :]])
    a = model(X)
    model.summary()
    # Now choosing batch size equal to number of training examples.
    model.fit(train_X, train_y, batch_size=batch_size, epochs=num_of_epoch)

    y_pred = model.predict(test_X)

    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)
    print(test_acc)


if __name__ == '__main__':
    main()