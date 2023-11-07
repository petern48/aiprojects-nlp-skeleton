import torch
import torch.nn as nn


class TransformerNetwork(torch.nn.Module):


    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x (tensor): the input to the model
        '''
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.reshape(-1, 1)

if __name__ == "__main__":
    model = TransformerNetwork()
    print(model(torch.randn(128,44,300)).shape)