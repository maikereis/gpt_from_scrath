import torch

@torch.no_grad()
def get_logits(model, idx, context_size):
    """
    Get the logits from the model for the last time step of the input sequence.

    Parameters:
    - model (nn.Module): The model to use for generating logits.
    - idx (torch.Tensor): The input tensor containing token indices.
    - context_size (int): The size of the context window.

    Returns:
    - torch.Tensor: The logits for the last time step.
    """
    idx_cond = idx[:, -context_size:]  # Crop to the last context_size tokens
    logits = model(idx_cond)  # Get model predictions
    return logits[:, -1, :]  # Return logits for the last time step


def filter_logits(logits, top_k):
    """
    Apply top-k filtering to the logits.

    Parameters:
    - logits (torch.Tensor): The logits to filter.
    - top_k (int): The number of top logits to keep.

    Returns:
    - torch.Tensor: The filtered logits.
    """
    top_logits, _ = torch.topk(logits, top_k)  # Get the top k logits
    min_val = top_logits[:, -1]  # Get the smallest of the top k logits
    # Replace logits below the top k threshold with -inf
    return torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)


def sample_next_token(logits, temperature, eos_id):
    """
    Sample the next token based on the logits and temperature settings.

    Parameters:
    - logits (torch.Tensor): The logits from which to sample.
    - temperature (float): The temperature for scaling logits.
    - eos_id (int): The end-of-sequence token ID.

    Returns:
    - torch.Tensor: The sampled token index.
    """
    if temperature > 0.0:
        logits = logits / temperature  # Scale logits by temperature
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the distribution
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Get the token with max logit

    return idx_next.squeeze(0)  # Remove unnecessary dimensions


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate a sequence of tokens using the provided model.

    This function orchestrates the generation process, calling the necessary
    helper functions to get logits, filter them, and sample the next token.

    Parameters are the same as previously defined.

    Returns:
    - torch.Tensor: A tensor containing the input tokens followed by the generated tokens.
    """

    if idx.size(0) == 0:
        raise ValueError("Input batch size cannot be zero.")

    for _ in range(max_new_tokens):
        logits = get_logits(model, idx, context_size)  # Get logits
        if top_k is not None:
            logits = filter_logits(logits, top_k)  # Filter logits if top_k is specified

        idx_next = sample_next_token(logits, temperature, eos_id)  # Sample the next token

        if idx_next == eos_id:  # Stop if end-of-sequence token is encountered
            break

        idx = torch.cat((idx, idx_next.unsqueeze(0)), dim=1)  # Append the next token to the sequence

    return idx  # Return the updated tensor with generated tokens
