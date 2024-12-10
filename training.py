import torch
from torch import tensor
import torch.nn as nn
import torch.utils.data
import torchmetrics
import random
import data_processing
import numpy as np
from livelossplot import PlotLosses
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import datetime


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
        dataset += [list(map(float, d[5:-1]))]
        target += [[0.] if d[4] == "D" else [1.]]
    data = data_set(dataset, target)
    return data

def accuracy(training_set, model):
    tensors = data_to_tensor(training_set)
    data = tensors.data
    labels = np.array(tensors.labels.tolist())
    results = model(data).tolist()
    results = np.array([[1] if x[0] > 0.5 else [0] for x in results])
    return np.sum(labels == results)/len(labels)
  




def train(nn, input_training_set, input_valid_set, h_params = {}, loss_fn = torch.nn.BCELoss()):
    
    training_set =  data_to_tensor(input_training_set)
    valid_set = data_to_tensor(input_valid_set)
    batch_size = h_params['Batch Size']
    epochs = h_params['Epochs']
    lr = h_params['LR']
    # Define the optimizer
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=lr)
    dataLoader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    print(type(nn.state_dict()))
    writer.add_text("Hyperparameters", str(h_params))
    beginning_time = str(datetime.datetime.now())[:-7]
    torch.save(nn.state_dict(), "./models/SD_" + beginning_time + ".pth")
    for epoch in range(epochs):
        for d, l in dataLoader:
  
            y_pred = nn(d)
            loss = loss_fn(y_pred, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_accuracy = accuracy(input_training_set, nn)
            valid_accuracy = accuracy(input_valid_set, nn)
            
            
            writer.add_scalar("Valid Accuracy", valid_accuracy, epoch)
            writer.add_scalar("Training Accuracy", training_accuracy, epoch)
            writer.add_scalar("Loss", loss, epoch)
            
            print('Epoch:', epoch, 'Loss:', np.round(loss.item(), 2), 'Training Set Accuracy:', np.round(training_accuracy, 3), 'Validation Accuracy', np.round(valid_accuracy, 3))
            torch.save(nn, "./models/PR_" + beginning_time + '.pth')
            # if test_accuracy > threshold:
            #     break
        # if test_accuracy> threshold:
        #     break
    return nn


def main():

    dict_list = data_processing.read_data("training_data_extended.csv")
    data = []
    for _ in dict_list:
        data += [list(_.values())]
    random.shuffle(data)
    valid_set = data[-4000:]
    test_set = data[:4000]
    training_set = data[4000:-4000]

    model = torch.nn.Sequential(
        torch.nn.Linear(96, 256, bias = True),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid(),
        
    )
    
    h_params = {"Epochs": 10000, "LR": 0.0025, "Batch Size": 4000}
    train(model, training_set, valid_set, h_params)
    writer.flush()
    writer.close()
if __name__ == "__main__":
    main()
    
