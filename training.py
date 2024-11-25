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
        target += [[0, 1] if d[3] == "D" else [1,0]]
    data = data_set(dataset, target)
    return data

def training_accuracy(training_set, model):
    tensors = data_to_tensor(training_set)
    data = tensors.data
    labels = np.array(tensors.labels.tolist())
    results = model(data).tolist()
    results = np.array([[1, 0] if x[0] > x[1] else [0, 1] for x in results])
    return np.sum((labels == results).all(axis = 1))/len(labels)




def train(nn, training_set, valid_set, loss_fn = torch.nn.MSELoss(), lr = 0.0002, batch_size = 10, epochs = 5):
    training_set =  data_to_tensor(training_set)
    valid_set = data_to_tensor(valid_set)
    # Define the optimizer
    optimizer = torch.optim.Adagrad(list(nn.parameters()), lr=lr)
    dataLoader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
   
    # Train the autoencoder
    for epoch in range(epochs):
        for d, l in dataLoader:
            # Forward pass
            y_pred = nn(d)

            # Compute the loss
            loss = loss_fn(y_pred.float(), l.float())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_accuracy = (torch.sum(torch.vmap(torch.argmax)(training_set.labels) == torch.vmap(torch.argmax)(nn(training_set.data)))/len(training_set.labels)).tolist()
            test_accuracy = np.round(test_accuracy,2)
            valid_accuracy = (torch.sum(torch.vmap(torch.argmax)(valid_set.labels) == torch.vmap(torch.argmax)(nn(valid_set.data)))/len(valid_set.labels)).tolist()
            valid_accuracy = np.round(valid_accuracy,2)
            print('Epoch:', epoch, 'Loss:', np.round(loss.item(), 2), 'Training Set Accuracy:', test_accuracy, 'Validation Accuracy:', valid_accuracy)
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
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 72),
        torch.nn.ReLU(),
        torch.nn.Linear(72, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 24),
        torch.nn.ReLU(),
        torch.nn.Linear(24, 2),
        torch.nn.Softmax()
    )


    nn_trained = train(model, training_set, valid_set, batch_size=10, epochs=100, lr = 0.025)

if __name__ == "__main__":
    main()

