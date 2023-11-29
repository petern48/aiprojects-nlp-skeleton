import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from tqdm.notebook import tqdm
# from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    model.train()

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    # loss_fn = nn.CrossEntropyLoss()  # more for multi-class classification
    loss_fn = nn.BCELoss()

    writer = SummaryWriter(filename_suffix=model.__class__.__name__)  # tensorboard log

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch_idx, batch in enumerate(tqdm(train_loader, disable=True)):  # show the times for each batch
            # Forward propagate
            samples, labels = batch['input_ids'].to(device), batch['labels'].to(device)

            # print("samples device ", samples.get_device())
            # print("labels device", labels.get_device())
            outputs = model(samples)

            labels = labels.reshape(-1,1).float()
            # Backpropagation and gradient descent
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # reset gradients before next iteration

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                print(f"Train Epoch {epoch} Loss {loss.item()}")
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * step, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                accuracy = compute_accuracy(outputs, labels)

                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar('Training Accuracy', accuracy, epoch)

                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
            step += 1

        with torch.no_grad():
            model.eval()
            val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device)
            model.train()
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        print(f"Validation Loss {val_loss}")
        print(f"Validation Accuracy {val_accuracy}")


        print()

    writer.close()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]
    Example output:
        0.75
    """
    # look into f1 score
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    for batch in val_loader:
        val_samples, val_labels = batch['input_ids'].to(device), batch['labels'].to(device)

        outputs = model(val_samples)
        val_labels = val_labels.reshape(-1, 1).float()
        val_loss = loss_fn(outputs, val_labels).item()  # change tensor to single val

        val_accuracy = compute_accuracy(outputs, val_labels)

    return val_loss, val_accuracy
