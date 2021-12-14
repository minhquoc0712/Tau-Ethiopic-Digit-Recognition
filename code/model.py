from torch.utils.data import Dataset
from torch.nn import Module, Conv2d, MaxPool2d, BatchNorm2d, Linear, Softmax
from torch.nn.functional import relu
from torch import LongTensor, from_numpy, tanh, argmax, mean, zeros
import torchvision.transforms as transforms
from math import floor


def first_dense_layer_dimesion(input_shape):
    dense_dim = [0, 0]
    for i in range(2):
        dense_dim[i] = floor((input_shape[i] + 2 * 0 - 4) / 1 + 1)
        dense_dim[i] = floor((dense_dim[i] + 2 * 0 - 2) / 2 + 1)
        dense_dim[i] = floor((dense_dim[i] + 2 * 0 - 5) / 1 + 1)
        dense_dim[i] = floor((dense_dim[i] + 2 * 0 - 3) / 3 + 1)

    return 800 * dense_dim[0] * dense_dim[1]


class DatasetTAU(Dataset):
    """
    Used to make data set with some additional properties.
    """
    def __init__(self, data_examples, labels):
        self.X = data_examples
        self.y = from_numpy(labels).type(LongTensor)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        example = self.X[index]
        label = self.y[index]

        return example, label


class MyModel(Module):
    def __init__(self, input_col1_shape, input_col2_shape, input_col3_shape, input_col4_shape,
                 input_col5_shape, num_col):
        super(MyModel, self).__init__()

        self.input_col1_shape = input_col1_shape
        self.input_col2_shape = input_col2_shape
        self.input_col3_shape = input_col3_shape
        self.input_col4_shape = input_col4_shape
        self.input_col5_shape = input_col5_shape

        self.num_col = num_col

        # First column.

        self.col1_conv2D1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))

        self.col1_batchnormalization1 = BatchNorm2d(20)
        self.col1_maxPool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.col1_conv2D2 = Conv2d(in_channels=20, out_channels=800, kernel_size=(5, 5), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col1_batchnormalization2 = BatchNorm2d(800)
        self.col1_maxPool2 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(1, 1))
        self.col1_linear1 = Linear(in_features=first_dense_layer_dimesion(input_col1_shape), out_features=150)
        self.col1_linear2 = Linear(in_features=150, out_features=10)
        self.col1_softmax = Softmax(dim=1)

        # Second column.
        self.col2_conv2D1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col2_batchnormalization1 = BatchNorm2d(20)
        self.col2_maxPool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.col2_conv2D2 = Conv2d(in_channels=20, out_channels=800, kernel_size=(5, 5), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col2_batchnormalization2 = BatchNorm2d(800)
        self.col2_maxPool2 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(1, 1))
        self.col2_linear1 = Linear(in_features=first_dense_layer_dimesion(input_col2_shape), out_features=150)
        self.col2_linear2 = Linear(in_features=150, out_features=10)
        self.col2_softmax = Softmax(dim=1)

        ###############################################################################################
        # Third column.
        self.col3_conv2D1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col3_batchnormalization1 = BatchNorm2d(20)
        self.col3_maxPool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.col3_conv2D2 = Conv2d(in_channels=20, out_channels=800, kernel_size=(5, 5), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col3_batchnormalization2 = BatchNorm2d(800)
        self.col3_maxPool2 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(1, 1))
        self.col3_linear1 = Linear(in_features=first_dense_layer_dimesion(input_col3_shape), out_features=150)
        self.col3_linear2 = Linear(in_features=150, out_features=10)
        self.col3_softmax = Softmax(dim=1)

        ###############################################################################################
        # Fourth column.
        self.col4_conv2D1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col4_batchnormalization1 = BatchNorm2d(20)
        self.col4_maxPool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.col4_conv2D2 = Conv2d(in_channels=20, out_channels=800, kernel_size=(5, 5), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col4_batchnormalization2 = BatchNorm2d(800)
        self.col4_maxPool2 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(1, 1))
        self.col4_linear1 = Linear(in_features=first_dense_layer_dimesion(input_col4_shape), out_features=150)
        self.col4_linear2 = Linear(in_features=150, out_features=10)
        self.col4_softmax = Softmax(dim=1)

        ###############################################################################################
        # Fifth column.
        self.col5_conv2D1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col5_batchnormalization1 = BatchNorm2d(20)
        self.col5_maxPool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.col5_conv2D2 = Conv2d(in_channels=20, out_channels=800, kernel_size=(5, 5), stride=(1, 1),
                                   padding=(0, 0), dilation=(1, 1))
        self.col5_batchnormalization2 = BatchNorm2d(800)
        self.col5_maxPool2 = MaxPool2d(kernel_size=(3, 3), stride=(3, 3), dilation=(1, 1))
        self.col5_linear1 = Linear(in_features=first_dense_layer_dimesion(input_col5_shape), out_features=150)
        self.col5_linear2 = Linear(in_features=150, out_features=10)
        self.col5_softmax = Softmax(dim=1)


    def forward(self, inputs):
        inputs = inputs.float()

        # First column.

        # Resize image.
        resize1 = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.input_col1_shape),
                                      transforms.ToTensor()])
        inputs_col1 = zeros((inputs.shape[0], 1, self.input_col1_shape[0], self.input_col1_shape[1]))
        for i in range(inputs.shape[0]):
            inputs_col1[i, 0] = resize1(inputs[i, 0])
        inputs_col1 = inputs_col1.double()

        output_col1 = self.col1_batchnormalization1(tanh(self.col1_conv2D1(inputs_col1)))
        output_col1 = relu(self.col1_maxPool1(output_col1))
        output_col1 = self.col1_batchnormalization2(tanh(self.col1_conv2D2(output_col1)))
        output_col1 = relu(self.col1_maxPool2(output_col1))
        output_col1 = output_col1.view(output_col1.shape[0], -1)
        output_col1 = tanh(self.col1_linear1(output_col1))
        output_col1 = self.col1_softmax(self.col1_linear2(output_col1))

        ##########################################################################################
        # Second column.

        # Resize image.
        resize2 = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.input_col2_shape),
                                      transforms.ToTensor()])
        inputs = inputs.float()
        inputs_col2 = zeros((inputs.shape[0], 1, self.input_col2_shape[0], self.input_col2_shape[1]))
        for i in range(inputs.shape[0]):
            inputs_col2[i, 0] = resize2(inputs[i, 0])
        inputs_col2 = inputs_col2.double()

        output_col2 = self.col2_batchnormalization1(tanh(self.col2_conv2D1(inputs_col2)))
        output_col2 = relu(self.col2_maxPool1(output_col2))
        output_col2 = self.col2_batchnormalization2(tanh(self.col2_conv2D2(output_col2)))
        output_col2 = relu(self.col2_maxPool2(output_col2))
        output_col2 = output_col2.view(output_col2.shape[0], -1)
        output_col2 = tanh(self.col2_linear1(output_col2))
        output_col2 = self.col2_softmax(self.col2_linear2(output_col2))

        #############################################################################################
        # Third column.

        # Resize image.
        resize3 = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.input_col3_shape),
                                      transforms.ToTensor()])
        inputs_col3 = zeros((inputs.shape[0], 1, self.input_col3_shape[0], self.input_col3_shape[1]))
        for i in range(inputs.shape[0]):
            inputs_col3[i, 0] = resize3(inputs[i, 0])
        inputs_col3 = inputs_col3.double()

        output_col3 = self.col3_batchnormalization1(tanh(self.col3_conv2D1(inputs_col3)))
        output_col3 = relu(self.col3_maxPool1(output_col3))
        output_col3 = self.col3_batchnormalization2(tanh(self.col3_conv2D2(output_col3)))
        output_col3 = relu(self.col3_maxPool2(output_col3))
        output_col3 = output_col3.view(output_col3.shape[0], -1)
        output_col3 = tanh(self.col3_linear1(output_col3))
        output_col3 = self.col3_softmax(self.col3_linear2(output_col3))

        #############################################################################################
        # Fourth column.

        # Resize image.
        resize4 = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.input_col4_shape),
                                      transforms.ToTensor()])
        inputs_col4 = zeros((inputs.shape[0], 1, self.input_col4_shape[0], self.input_col4_shape[1]))
        for i in range(inputs.shape[0]):
            inputs_col4[i, 0] = resize4(inputs[i, 0])
        inputs_col4 = inputs_col4.double()

        output_col4 = self.col4_batchnormalization1(tanh(self.col4_conv2D1(inputs_col4)))
        output_col4 = relu(self.col4_maxPool1(output_col4))
        output_col4 = self.col4_batchnormalization2(tanh(self.col4_conv2D2(output_col4)))
        output_col4 = relu(self.col3_maxPool2(output_col4))
        output_col4 = output_col4.view(output_col4.shape[0], -1)
        output_col4 = tanh(self.col4_linear1(output_col4))
        output_col4 = self.col4_softmax(self.col3_linear2(output_col4))

        #############################################################################################
        # Fifth column. Open for used fifth column.
        resize5 = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(self.input_col5_shape),
                                          transforms.ToTensor()])

        inputs_col5 = zeros((inputs.shape[0], 1, self.input_col5_shape[0], self.input_col5_shape[1]))
        for i in range(inputs.shape[0]):
            inputs_col5[i, 0] = resize5(inputs[i, 0])
        inputs_col5 = inputs_col5.double()

        output_col5 = self.col5_batchnormalization1(tanh(self.col5_conv2D1(inputs_col5)))
        output_col5 = relu(self.col5_maxPool1(output_col5))
        output_col5 = self.col5_batchnormalization2(tanh(self.col5_conv2D2(output_col5)))
        output_col5 = relu(self.col5_maxPool2(output_col5))
        output_col5 = output_col5.view(output_col5.shape[0], -1)

        output_col5 = tanh(self.col5_linear1(output_col5))
        output_col5 = self.col4_softmax(self.col3_linear2(output_col5))

        return (output_col1 + output_col2 + output_col3 + output_col4 + output_col5) / 5

