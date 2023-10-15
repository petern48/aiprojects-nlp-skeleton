import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter()  # tensorboard log

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):  # show the times for each batch
            # Forward propagate
            samples, labels = batch[0], batch[1]

            samples.to(device)
            labels.to(device)
            outputs = model(samples)


            # Backpropagation and gradient descent
            loss = loss_fn(outputs, labels)
            loss.backwards()
            optimizer.step()
            optimizer.zero_grads()  # reset gradients before next iteration


            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                with torch.no_grad():
                    accuracy = compute_accuracy(outputs, labels)

                    writer.add_scalar('Training Loss', loss, epoch)
                    writer.add_scalar('Training Accuracy', accuracy, epoch)


                    # TODO:
                    # Compute validation loss and accuracy.
                    # Log the results to Tensorboard.
                    # Don't forget to turn off gradient calculations!
                    val_loss, val_accuracy = evaluate(val_loader, model, loss_fn)
                    writer.add_scalar('Validation Loss', val_loss, epoch)
                    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

            step += 1

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

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    for batch in tqdm(val_loader):
        val_samples, val_labels = batch
        outputs = model(val_samples)
        val_loss = loss_fn(outputs, val_labels)

        val_accuracy = compute_accuracy(outputs, val_labels)

    return val_loss, val_accuracy
