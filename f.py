import numpy as np
from input_and_output import read_test_data
from torch import load
from model import MyModel
from torch import from_numpy, argmax

path_test = "tau-ethiopic-digit-recognition/test"
test_data = read_test_data(10000, path_test)
test_data = from_numpy(test_data)

model = MyModel(input_col1_shape=(28, 26), input_col2_shape=(28, 22),
                    input_col3_shape=(28, 20), input_col4_shape=(28, 18),
                    input_col5_shape=(28, 28), num_col=5)

model.load_state_dict(load('Assignment_150'))
model.double()

model.eval()
y_pred = np.zeros((test_data.shape[0]))
for i in range(100):
    y = model(test_data[(i * 100):((i + 1) * 100), :, :, :])
    y = argmax(y, dim=1)
    y_pred[(i * 100):((i + 1) * 100)] = y.numpy()

y_pred = y_pred.astype('int16')

text_file = open('result.txt', 'w')
text_file.write('Id,Category\n')
for i in range(y_pred.shape[0]):
    if i % 100 == 0:
        print(i)
    text_file.write("{:05d},".format(i) + "{:d}\n".format(y_pred[i] + 1))
text_file.close()
