import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data
import torchmetrics
import random
import data_processing
import numpy as np

class data_set(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_nn():
    # Return the encoder network
    return torch.nn.Sequential(
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 2),
        torch.nn.Softmax()
    )


def data_to_tensor(training_set):
    target = []
    dataset = []
    for d in training_set:
        dataset += [list(map(float, d[4:]))]
        target += [[0.] if d[3] == "D" else [1.]]
    data = data_set(dataset, target)
    return data

def training_accuracy(training_set, model):
    tensors = data_to_tensor(training_set)
    data = tensors.data
    labels = np.array(tensors.labels.tolist())
    results = model(data).tolist()
    results = np.array([[1] if x[0] > 0.5 else [0] for x in results])
    return np.sum(labels == results)/len(labels)
  




def train(nn, input_training_set, input_valid_set, loss_fn = torch.nn.BCELoss(), lr = 0.0002, batch_size = 10, epochs = 5, threshold = 0.9):
    training_set =  data_to_tensor(input_training_set)
    valid_set = data_to_tensor(input_valid_set)
    # Define the optimizer
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)
    dataLoader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
   
    # Train the autoencoder
    for epoch in range(epochs):
        for d, l in dataLoader:
            # Forward pass
            y_pred = nn(d)

            # Compute the loss
            loss = loss_fn(y_pred, l)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_accuracy = training_accuracy(input_training_set, nn)
            test_accuracy = np.round(test_accuracy,2)
            valid_accuracy = training_accuracy(input_valid_set, nn)
            valid_accuracy = np.round(valid_accuracy, 2)
            print('Epoch:', epoch, 'Loss:', np.round(loss.item(), 2), 'Training Set Accuracy:', test_accuracy, 'Validation Accuracy', valid_accuracy)
            if valid_accuracy > threshold:
                break
        if valid_accuracy> threshold:
            break
        torch.save(nn, 'model.pth')

    return nn


def main():
    dict_list = data_processing.read_data("training_data_long.csv")
    data = []
    for _ in dict_list:
        data += [list(_.values())[1:]]
    random.shuffle(data)
    valid_set = data[-1000:]
    test_set = data[:1000]
    training_set = data[1000:-1000]


    model = torch.nn.Sequential(
        torch.nn.Linear(72, 256, bias = True),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid(),
    )

    nn = train(model, training_set, valid_set, batch_size= int(len(training_set)/1), epochs=3000, lr = 0.005, threshold=0.93)
    #model(data_to_tensor(training_set).data), data_to_tensor(training_set).labels, data_to_tensor(training_set).data


if __name__ == "__main__":
    main()
    
