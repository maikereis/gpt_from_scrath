import torch
from utils import generate_and_print_sample
from torcheval.metrics.functional import binary_precision

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a batch of input data and target labels using a given model.

    This function takes a batch of input data and target labels, performs a forward pass
    through the model, and computes the loss using cross-entropy.

    Parameters:
    - input_batch (torch.Tensor): A tensor containing the input data for the model.
    - target_batch (torch.Tensor): A tensor containing the target labels corresponding to the input data.
    - model (torch.nn.Module): The model used to make predictions.
    - device (torch.device): The device (CPU or GPU) on which the model and data are located.

    Returns:
    - torch.Tensor: The computed loss as a scalar tensor.
    """
    model.eval()  # Ensure model is in evaluation mode

    # Move tensors to the specified device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass without tracking gradients
    logits = model(input_batch)[:, -1, :]

    # Compute loss
    loss = torch.nn.functional.cross_entropy(logits, target_batch)

    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss over a specified number of batches from a data loader.

    This function iterates through a data loader, computes the loss for each batch using
    a given model, and returns the average loss. If the data loader is empty, it returns NaN.

    Parameters:
    - data_loader (torch.utils.data.DataLoader): The data loader providing input-target pairs.
    - model (torch.nn.Module): The model used to compute predictions and loss.
    - device (torch.device): The device (CPU or GPU) on which the model and data are located.
    - num_batches (int, optional): The number of batches to process. If None, processes all batches.

    Returns:
    - float: The average loss over the processed batches. Returns NaN if the data loader is empty.
    """
    total_loss = 0.0

    data_loader_length = len(data_loader)  # Store the length of the data loader

    if data_loader_length == 0:
        return float("nan")

    # Determine the number of batches to process
    num_batches = num_batches if num_batches is not None else data_loader_length
    num_batches = min(num_batches, data_loader_length)

    # Iterate through the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # Accumulate loss
        else:
            break

    # Return average loss over the number of batches processed
    return total_loss / num_batches


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model on training and validation datasets.

    This function calculates the average loss for both the training and validation datasets
    using a specified number of iterations. The model is switched to evaluation mode during
    the evaluation process and switched back to training mode afterward.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): Data loader for the validation dataset.
    - device (torch.device): The device (CPU or GPU) on which the model and data are located.
    - eval_iter (int): The number of batches to evaluate for loss computation.

    Returns:
    - tuple: A tuple containing the average training loss and average validation loss.
    """
    model.eval()
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Get logits
            predicted_labels = torch.argmax(logits, dim=-1)  # Predicted classes

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples if num_examples > 0 else 0.0

def calc_precision_loader(data_loader, model, device, num_batches=None):
    model.eval()
    y_true = []
    y_pred = []

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Get logits
            predicted_labels = torch.argmax(logits, dim=-1)  # Predicted classes

            y_pred.append(predicted_labels)
            y_true.append(target_batch)
        else:
            break

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Calculate precision
    precision = binary_precision(y_true, y_pred).item()

    return precision


def log_training_evaluation(
    model,
    train_loader,
    val_loader,
    device,
    eval_iter,
    metrics,
    examples_seen,
    global_step,
    epoch,
):
    """Logs training and validation losses."""
    train_loss, val_loss = evaluate_model(
        model, train_loader, val_loader, device, eval_iter
    )

    metrics["train_losses"].append(train_loss)
    metrics["val_losses"].append(val_loss)

    print(
        f"Epoch {epoch + 1} (Step {global_step:06d}): "
        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
    )

def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "train_precision": [],
        "val_precision": [],
    }

    examples_seen = 0
    global_step = -1  # Initialize global step counter

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients for the optimizer
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                log_training_evaluation(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    metrics,
                    examples_seen,
                    global_step,
                    epoch,
                )

        train_precision = calc_precision_loader(train_loader, model, device, num_batches = eval_iter)
        val_precision = calc_precision_loader(val_loader, model, device, num_batches = eval_iter)

        metrics['train_precision'].append(train_precision)
        metrics['val_precision'].append(val_precision)

        print(f"Training precision: {train_precision * 100:.2f}% | ", end ="")
        print(f"Validation precision: {val_precision * 100:.2f}%")


    return metrics['train_losses'], metrics["val_losses"], metrics['train_precision'], metrics['val_precision'], examples_seen
