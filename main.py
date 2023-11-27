import os
import sys

import constants
from data.Dataset import Dataset
from data.GetEmbeddings import getEmbeddings
from networks.StartingNetwork import BaseNetwork
from networks.transformer import Transformer, load_transformer_model, save_transformer_model
from train_functions.starting_train import starting_train, evaluate
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from datetime import datetime
from load_args import load_args

def main():

    args = load_args()
    # 1st arg is pretrained model path
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set as {device}")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    embeddings_path = 'glove.6B.50d.txt'
    if args.embs_path:
        embeddings_path = args.embs_path
    pad_token = '<pad>'
    unk_token = '<unk>'
    max_seq_length = 134
    vocab_npa, embs_npa = getEmbeddings(embeddings_path, pad_token, unk_token)
    embs_dim = embs_npa.shape[1]

    # Initalize dataset and model.
    data_path = "dev.csv"
    if args.data_file:
        data_path = args.data_file
    df = pd.read_csv(data_path)
    x = df['question_text'].array  # turn into array to remove the randomized indexing of pd.Series
    y = df['target'].array
    # split will be consistent across multiple rules
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    train_dataset = Dataset(x_train, y_train, vocab_npa, pad_token, unk_token)
    val_dataset = Dataset(x_test, y_test, vocab_npa, pad_token, unk_token)

    # # TODO Load pretrained model
    if args.pretrained_model:
        # load
        model = load_transformer_model(args.pretrained_model, embs_npa)
        model.to(device)
        print("pretrained model loaded")

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
            device=device,
            using_notebook=args.using_notebook
        )

        # Create pretrained directory if not yet created
        if not os.path.isdir(constants.PRETRAINED_DIR):
            os.mkdir(constants.PRETRAINED_DIR)

        now = datetime.now()
        datetime_str = now.strftime("%m-%d-%H-%M-%S")
        model_save_path = os.path.join(
            constants.PRETRAINED_DIR,
            f'{datetime_str}-{model.__class__.__name__}-model-{num_layers}-layers-{num_heads}-heads-{constants.EPOCHS}-epochs.pt'
        )
        print('model_save_path', model_save_path)
        save_transformer_model(model_save_path, model)
        print(f"model saved at {datetime_str}")

    # inference, evaluate model
    print("evaluating model")
    model.eval()

    # TODO: temporarily just use val_dataset
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=constants.BATCH_SIZE, shuffle=True
    )

    loss_fn = torch.nn.BCELoss()
    test_loss, test_accuracy = evaluate(test_loader, model, loss_fn, device)

    print("Final test_loss: ", test_loss)
    print("Final test_accuracy ", test_accuracy)


if __name__ == "__main__":
    main()
