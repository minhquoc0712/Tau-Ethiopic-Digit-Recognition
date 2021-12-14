from matplotlib import image
from model import *
import numpy as np


def accuracy(predictions, labels):
    classes = argmax(predictions, dim=1)
    return mean((classes == labels).float())


def read_train_data(n, path):
    train_X = np.zeros((10 * n, 28, 28))
    train_y = np.zeros((10 * n, 1))

    for i in range(10):
        print(f'Reading training data from class {i + 1}')

        # Reading image in each class.
        for j in range(n):
            train_X[i * n + j, :, :] = image.imread(f"{path}/{i + 1}/" + "{:05d}.jpg".format(j + 1))
            train_y[i * n + j] = i

    train_X = np.expand_dims(train_X, axis=1).astype('double')

    return train_X, train_y


def read_train_data_as_text(n, path):
    train_X = np.zeros((10 * n, 28, 28))
    train_y = np.zeros((10 * n, 1))

    for i in range(10):
        print(f'Reading training data from class {i + 1}')

        # Reading image in each class.
        for j in range(n):
            train_X[i * n + j, :, :] = image.imread(f"{path}/{i + 1}/00" + "{:03d}.jpg".format(j + 1))

            train_y[i * n + j] = i

    train_X = np.expand_dims(train_X, axis=1).astype('double')

    return train_X, train_y


def read_test_data(m, path):
    test_X = np.zeros((m, 28, 28))

    for i in range(m):
        test_X[i, :, :] = image.imread(f"{path}/" + "{:05d}.jpg".format(i))

    test_X = np.expand_dims(test_X, axis=1).astype('double')
    return test_X


def normalize(trdata, X):
    trdata_norm = trdata.astype('float32')
    X_norm = X.astype('float32')
    trdata_norm = trdata_norm/255.0
    X_norm = X_norm/255.0
    return trdata_norm, X_norm



def evaluate(model, dataloader):
    """
    Evaluate model accuracy with some train data examples.
    :param model:
    :param dataloader:
    :return:
    """
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summary = 0
    prediction = np.zeros((len(dataloader), 1))

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        # compute model output
        output_batch = model(data_batch)
        output_batch = squeeze(output_batch, 1)

        predict_batch = output_batch.data

        y_pred = argmax(predict_batch, dim=1, keepdim=True)
        prediction[i * predict_batch.shape[0]:(i + 1) * predict_batch.shape[0]] = y_pred

        labels_batch = labels_batch.data

        # compute all metrics on this batch
        summary = accuracy(predict_batch, labels_batch)

    return summary, y_pred
