import os
import sys

import constants
from data.StartingDataset import StartingDataset
from data.GetEmbeddings import getEmbeddings
from networks.StartingNetwork import BaseNetwork
from train_functions.starting_train import starting_train
import torch


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    data_path = "dev.csv"  # TODO: make sure you have train.csv downloaded in your project! this assumes it is in the project's root directory (ie the same directory as main) but you can change this as you please
    embeddings_path = 'glove.6B.50d.txt'
    pad_token = '<pad>'
    unk_token = '<unk>'
    max_seq_length = 134

    vocab_npa, embs_npa = getEmbeddings(embeddings_path, pad_token, unk_token)

    train_dataset = StartingDataset(data_path, vocab_npa, pad_token, unk_token)
    val_dataset = StartingDataset(data_path, vocab_npa, pad_token, unk_token)

    model = BaseNetwork(embs_npa, max_seq_length, device)
    model.to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
