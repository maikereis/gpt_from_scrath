import torch

def text_to_token_ids(text, tokenizer, device=None):
    """
    Convert a text string to a tensor of token IDs using the specified tokenizer.

    This function tokenizes the input text and creates a PyTorch tensor from the token IDs,
    optionally moving the tensor to a specified device (e.g., CPU or GPU).

    Parameters:
    - text (str): The input text to be tokenized.
    - tokenizer (Tokenizer): The tokenizer object used to encode the text into token IDs.
    - device (torch.device, optional): The device to which the tensor will be moved. If None,
                                       the tensor will be created on the CPU.

    Returns:
    - torch.Tensor: A tensor containing the token IDs of the input text, with an added batch dimension.
    """
    # Tokenize the text (assuming tokenizer.encode returns a list of ids)
    encoded = tokenizer.encode(text)
    # Create tensor and move to the device in one step
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)  # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert a tensor of token IDs back to a text string using the specified tokenizer.

    This function removes the batch dimension from the tensor of token IDs and decodes
    it back into a human-readable text string.

    Parameters:
    - token_ids (torch.Tensor): A tensor containing token IDs, expected to have a batch size of 1.
    - tokenizer (Tokenizer): The tokenizer object used to decode the token IDs back to text.

    Returns:
    - str: The decoded text string corresponding to the input token IDs.
    """
    # Remove batch dimension, decode token ids to text
    flat = token_ids.squeeze(0)  # Assumes batch size is 1
    return tokenizer.decode(flat.tolist())
