import torch
from utils import generate_and_print_sample

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
    logits = model(input_batch)

    # Compute loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1)
    )

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


def log_training_evaluation(model, train_loader, val_loader, device, eval_iter, metrics, tokens_seen, global_step, epoch):
    """Logs training and validation losses."""
    train_loss, val_loss = evaluate_model(
        model, train_loader, val_loader, device, eval_iter)

    metrics['train_losses'].append(train_loss)
    metrics['val_losses'].append(val_loss)
    metrics['tokens_seen'].append(tokens_seen)

    print(f"Epoch {epoch + 1} (Step {global_step:06d}): "
          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Train a model on the training dataset and evaluate it on the validation dataset.

    This function performs the training of the specified model for a defined number of epochs,
    computes losses for both training and validation datasets, and logs relevant metrics.
    It also generates sample outputs after each epoch.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): Data loader for the validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used to update model weights.
    - device (torch.device): The device (CPU or GPU) on which the model and data are located.
    - num_epochs (int): The total number of epochs for training.
    - eval_freq (int): Frequency of evaluation (in terms of global steps) during training.
    - eval_iter (int): Number of batches to evaluate for loss computation.
    - start_context (str): Initial text context for text generation.
    - tokenizer: The tokenizer used to convert text to tokens and vice versa.

    Returns:
    - dict: A dictionary containing lists of training losses, validation losses, and tokens seen.
    """
    # Initialize a dictionary to track losses and tokens seen
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'tokens_seen': [],
    }

    tokens_seen = 0
    global_step = -1 # Initialize global step counter

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients for the optimizer
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model weights

            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluate model periodically
            if global_step % eval_freq == 0:
                log_training_evaluation(model, train_loader, val_loader,
                                         device, eval_iter, metrics, tokens_seen, global_step, epoch)

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return metrics
