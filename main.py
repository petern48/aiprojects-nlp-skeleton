import os
import sys

import constants
from data.Dataset import Dataset
from data.GetEmbeddings import getEmbeddings
from networks.StartingNetwork import BaseNetwork
from networks.transformer import Transformer
from train_functions.starting_train import starting_train
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import sys


def main():
    # 1st arg is pretrained model path
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    embeddings_path = 'glove.6B.50d.txt'
    pad_token = '<pad>'
    unk_token = '<unk>'
    max_seq_length = 134
    vocab_npa, embs_npa = getEmbeddings(embeddings_path, pad_token, unk_token)
    embs_dim = embs_npa.shape[1]

    # Initalize dataset and model.
    data_path = "dev.csv"
    df = pd.read_csv(data_path)
    x = df['question_text'].array  # turn into array to remove the randomized indexing of pd.Series
    y = df['target'].array
    # split will be consistent across multiple rules
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    train_dataset = Dataset(x_train, y_train, vocab_npa, pad_token, unk_token)
    val_dataset = Dataset(x_test, y_test, vocab_npa, pad_token, unk_token)

    # # TODO Load pretrained model
    if len(sys.argv) != 1:
        pass
    #     model = torch.load(sys.argv[1])


    # Train new model
    else:
        # model = BaseNetwork(embs_npa, max_seq_length, device)
        num_heads = 5
        num_layers = 6
        d_model = embs_dim
        assert(embs_dim % num_heads == 0)
        model = Transformer(d_model, num_layers, num_heads, embs_npa)
        model.to(device)

        starting_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            hyperparameters=hyperparameters,
            n_eval=constants.N_EVAL,
            device=device
        )

        model_save_path = f'{model.__class__.__name__}-model-{num_layers}-n_layers-{num_heads}-heads-{constants.EPOCHS}-epochs.pt'
        torch.save(model.state_dict(), model_save_path)

    # inference, evaluate model
    model.eval()


if __name__ == "__main__":
    main()
