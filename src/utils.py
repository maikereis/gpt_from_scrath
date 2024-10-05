import torch
from src.token import token_ids_to_text, text_to_token_ids
from src.generate import get_logits

@torch.no_grad()  # Apply no_grad as a decorator
def generate_sequence(model, idx, max_new_tokens, context_size):
    """
    Generate a sequence of token indices using a language model.

    This function takes the current context (input token indices) and iteratively
    generates new tokens by predicting the next token based on the model's output.

    Parameters:
    - model (torch.nn.Module): The language model used for generating predictions.
    - idx (torch.Tensor): A tensor of shape (B, T) containing the current context,
                          where B is the batch size and T is the number of tokens.
    - max_new_tokens (int): The maximum number of new tokens to generate.
    - context_size (int): The maximum number of tokens from the context to consider
                          for generating the next token.

    Returns:
    - torch.Tensor: A tensor of shape (B, T + max_new_tokens) containing the input
                    context with the newly generated tokens appended.
    """
    model.eval()  # Switch to evaluation mode once

    if idx.size(0) == 0:
        raise ValueError("Input batch size cannot be zero.")

    for _ in range(max_new_tokens):
        logits = get_logits(model, idx, context_size)

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate a sample of text from a language model and print it.

    This function takes an initial context, tokenizes it, generates new tokens using the model,
    and then decodes and prints the resulting text.

    Parameters:
    - model (torch.nn.Module): The language model used for text generation.
    - tokenizer: The tokenizer used for encoding and decoding text.
    - device (torch.device): The device (CPU or GPU) on which the model and data reside.
    - start_context (str): The initial text context to start generating from.

    Returns:
    - None: This function prints the generated text directly to the console.
    """
    # Get context size based on model's positional embeddings
    context_size = model.pos_emb.weight.shape[0]

    # Tokenize input context
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # Generate text (no_grad is already handled in generate_text_simple)
    token_ids = generate_sequence(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
    )

    # Decode and print the text
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
