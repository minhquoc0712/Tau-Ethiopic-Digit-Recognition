from torch.utils.data import DataLoader
from input_and_output import read_train_data, accuracy
from model import DatasetTAU, MyModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import save, squeeze


path_train = "tau-ethiopic-digit-recognition/train"
train_X, train_y = read_train_data(6000, path_train)
training_data = DatasetTAU(train_X, train_y)
training_generator = DataLoader(training_data, batch_size=100, shuffle=True)

# Construct our model by instantiating the class defined above
model = MyModel(input_col1_shape=(28, 26), input_col2_shape=(28, 22),
                    input_col3_shape=(28, 20), input_col4_shape=(28, 18),
                    input_col5_shape=(28, 28), num_col=5)

model.double()

num_of_epoch = 1000
criterion = CrossEntropyLoss()
learning_rate = 0.001 / 0.993

for epoch in range(num_of_epoch):

    # Training
    i = 0
    running_loss = 0
    running_accuracy = 0
    LEARNING_RATE = max(0.00003, (learning_rate * 0.993))
    for local_batch, local_labels in training_generator:

        if i % 100 == 0:
            print(i)
        i += 1

        local_labels = squeeze(local_labels, 1)

        # zero the parameter gradients
        optimizer = Adam(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()

        y_pred = model(local_batch)
        loss = criterion(y_pred, local_labels)
        loss.backward()
        optimizer.step()

        # print(argmax(y_pred, dim=1), local_labels)

        running_loss += loss.item()
        running_accuracy += accuracy(y_pred, local_labels)

    save(model.state_dict(), f'Assignment_{epoch + 1}')

    running_loss /= len(training_generator)
    running_accuracy /= len(training_generator)

    print("\nEpoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, num_of_epoch,
                                                       running_loss, running_accuracy))