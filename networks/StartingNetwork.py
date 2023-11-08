import torch
import torch.nn as nn
import sys


class BaseNetwork(torch.nn.Module):
    """
    Basic logistic regression example. You may need to double check the dimensions :)
    """

    def __init__(self, embs_npa, device, freeze_embeddings=True):
        super().__init__()
    
        if freeze_embeddings:
            print('Freezing embeddings layer')


        self.embedding_layer = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(), 
            freeze=freeze_embeddings
        )
        # might need super(BaseNetwork, self).__init__()
        self.vocab_size = embs_npa.shape[0]
        # print('VOCAB SIZE', self.vocab_size)
        self.embedding_dim = embs_npa.shape[1]
        # print('EMBEDDING_DIM ', self.embedding_dim)
        self.device = device

        nn.Sequential(
            self.embedding_layer, 
            torch.nn.Dropout(p=0.2), 
            
        )


        # freeze embeddings layer


        # mid_dim = 134 * self.embedding_dim  # just for testing and trying to get things to work. 134 is max_seq_length
        # print('mid_dim ', mid_dim)  # number of samples 50
        self.fc1 = nn.Linear(self.embedding_dim, 50)
        self.fc2 = nn.Linear(50, 1)  # last layer needs to have output dim 1
        # [32, 134] dim
        # flatten in forward
        self.fc3 = nn.Linear(134, 1)

        # f1 score instead of accuracy

        self.sigmoid = nn.Sigmoid()
        self.relu = torch.nn.functional.relu


    def forward(self, input_ids): #, seq_length):
        '''
        input_ids (tensor): the input to the model
        '''

        # x = input_ids.to(self.device)  # moved to gpu in training loop
        # input_ids are ints
        #print('input shape ', input_ids.shape)  # [32,134]
        embeds = self.embedding_layer(input_ids)
        #print('EMBEDS output SHAPE', embeds.shape)  # [32, 134, 50]  # dims of embeddings 50d
        #print(self.vocab_size, 'vocab')
        #print(self.embedding_dim, 'embedding dim')
        # print()
        # sys.exit()
        # .reshape(-1. 134 * 50)
        # embeds = embeds.reshape(-1, 50)

        # x = self.fc1(x.to(torch.float32))  # original code
        # embeds are floats
        x = self.fc1(embeds)
        x = self.fc2(x)
        x = x.flatten(-2, -1)  # or squeeze
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
        # return x.reshape(-1, 1)  # original code
