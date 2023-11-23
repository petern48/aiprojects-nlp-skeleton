import torch
import torch.nn as nn
import sys


class BaseNetwork(torch.nn.Module):
    """
    """

    def __init__(self, embs_npa, device, freeze_embeddings=True):
        super().__init__()
        # might need super(BaseNetwork, self).__init__()
        self.vocab_size = embs_npa.shape[0]
        self.embedding_dim = embs_npa.shape[1]
        self.device = device

        # freeze embeddings layer
        if freeze_embeddings:
            print('Freezing embeddings layer')


        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(), 
            freeze=freeze_embeddings
        )

        # [32, 134, 50]
        self.fc1 = nn.Linear(self.embedding_dim, 50)
        self.fc2 = nn.Linear(50, 1)  # last layer needs to have output dim 1
        # [32, 134, 1]
        # flatten in forward
        # [32, 134] dim
        self.fc3 = nn.Linear(134, 1)
        # [32, 1]

        # f1 score instead of accuracy

        self.sigmoid = nn.Sigmoid()
        self.relu = torch.nn.functional.relu


    def forward(self, input_ids): #, seq_length):
        '''
        input_ids (tensor): the input to the model
        '''

        # x = input_ids.to(self.device)  # moved to gpu in training loop
        # input_ids are ints
        # print('input shape ', input_ids.shape)  # [32,134]
        embeds = self.embedding_layer(input_ids)

        # embeds are floats
        x = self.fc1(embeds)
        x = self.fc2(x)
        x = x.flatten(-2, -1)  # or squeeze
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
