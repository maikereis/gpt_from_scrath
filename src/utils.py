import torch
from collections import Counter
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
        model=model, idx=encoded, max_new_tokens=50, context_size=context_size
    )

    # Decode and print the text
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format


def __assign_check(left, right):
    """
    Checks if the shapes of two tensors match and assigns the right tensor to a new parameter if they do.

    This function compares the shapes of two tensors. If the shapes match, it creates a new
    parameter from the right tensor. If the shapes do not match, it raises a ValueError.

    Parameters:
    - left (torch.Tensor): The first tensor to compare.
    - right (torch.Tensor): The second tensor to compare and assign.

    Returns:
    - torch.nn.Parameter: A new parameter containing a detached clone of the right tensor.

    Raises:
    - ValueError: If the shapes of the two tensors do not match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_weights(gpt, gpt_hf, model_configs):
    """
    Load weights from a Hugging Face GPT model into a custom GPT model.

    This function transfers the weights from a pre-trained Hugging Face GPT model to a custom GPT model,
    ensuring that the shapes of the weights match and assigning them appropriately.

    Parameters:
    - gpt (torch.nn.Module): The custom GPT model to which the weights will be assigned.
    - gpt_hf (torch.nn.Module): The pre-trained Hugging Face GPT model from which the weights will be loaded.
    - model_configs (dict): A dictionary containing the model configuration, including the number of layers.

    Returns:
    - None
    """
    d = gpt_hf.state_dict()

    gpt.pos_emb.weight = __assign_check(gpt.pos_emb.weight, d["wpe.weight"])
    gpt.tok_emb.weight = __assign_check(gpt.tok_emb.weight, d["wte.weight"])

    for b in range(model_configs["n_layers"]):
        q_w, k_w, v_w = torch.split(
            d[f"h.{b}.attn.c_attn.weight"],
            d[f"h.{b}.attn.c_attn.weight"].size(-1) // 3,
            dim=-1,
        )
        gpt.transformer_decoders[b].multi_head_attention.W_query.weight = (
            __assign_check(
                gpt.transformer_decoders[b].multi_head_attention.W_query.weight, q_w.T
            )
        )
        gpt.transformer_decoders[b].multi_head_attention.W_key.weight = __assign_check(
            gpt.transformer_decoders[b].multi_head_attention.W_key.weight, k_w.T
        )
        gpt.transformer_decoders[b].multi_head_attention.W_value.weight = (
            __assign_check(
                gpt.transformer_decoders[b].multi_head_attention.W_value.weight, v_w.T
            )
        )

        q_b, k_b, v_b = torch.split(
            d[f"h.{b}.attn.c_attn.bias"],
            d[f"h.{b}.attn.c_attn.bias"].size(-1) // 3,
            dim=-1,
        )
        gpt.transformer_decoders[b].multi_head_attention.W_query.bias = __assign_check(
            gpt.transformer_decoders[b].multi_head_attention.W_query.bias, q_b
        )
        gpt.transformer_decoders[b].multi_head_attention.W_key.bias = __assign_check(
            gpt.transformer_decoders[b].multi_head_attention.W_key.bias, k_b
        )
        gpt.transformer_decoders[b].multi_head_attention.W_value.bias = __assign_check(
            gpt.transformer_decoders[b].multi_head_attention.W_value.bias, v_b
        )

        gpt.transformer_decoders[b].multi_head_attention.out_proj.weight = (
            __assign_check(
                gpt.transformer_decoders[b].multi_head_attention.out_proj.weight,
                d[f"h.{b}.attn.c_proj.weight"].T,
            )
        )
        gpt.transformer_decoders[b].multi_head_attention.out_proj.bias = __assign_check(
            gpt.transformer_decoders[b].multi_head_attention.out_proj.bias,
            d[f"h.{b}.attn.c_proj.bias"],
        )

        gpt.transformer_decoders[b].feed_foward.layers[0].weight = __assign_check(
            gpt.transformer_decoders[b].feed_foward.layers[0].weight,
            d[f"h.{b}.mlp.c_fc.weight"].T,
        )
        gpt.transformer_decoders[b].feed_foward.layers[0].bias = __assign_check(
            gpt.transformer_decoders[b].feed_foward.layers[0].bias,
            d[f"h.{b}.mlp.c_fc.bias"],
        )
        gpt.transformer_decoders[b].feed_foward.layers[2].weight = __assign_check(
            gpt.transformer_decoders[b].feed_foward.layers[2].weight,
            d[f"h.{b}.mlp.c_proj.weight"].T,
        )
        gpt.transformer_decoders[b].feed_foward.layers[2].bias = __assign_check(
            gpt.transformer_decoders[b].feed_foward.layers[2].bias,
            d[f"h.{b}.mlp.c_proj.bias"],
        )

        gpt.transformer_decoders[b].norm_layer1.weight = __assign_check(
            gpt.transformer_decoders[b].norm_layer1.weight, d[f"h.{b}.ln_1.weight"]
        )
        gpt.transformer_decoders[b].norm_layer1.bias = __assign_check(
            gpt.transformer_decoders[b].norm_layer1.bias, d[f"h.{b}.ln_1.bias"]
        )

        gpt.transformer_decoders[b].norm_layer2.weight = __assign_check(
            gpt.transformer_decoders[b].norm_layer2.weight, d[f"h.{b}.ln_2.weight"]
        )
        gpt.transformer_decoders[b].norm_layer2.bias = __assign_check(
            gpt.transformer_decoders[b].norm_layer2.bias, d[f"h.{b}.ln_2.bias"]
        )

        gpt.final_norm.weight = __assign_check(gpt.final_norm.weight, d[f"ln_f.weight"])
        gpt.final_norm.bias = __assign_check(gpt.final_norm.bias, d[f"ln_f.bias"])
        gpt.out.weight = __assign_check(gpt.out.weight, d["wte.weight"])


def value_counts(data_loader, normalize=False):
    """
    Count the occurrences of each label in a PyTorch DataLoader, similar to pd.value_counts().

    Args:
        data_loader (DataLoader): A PyTorch DataLoader instance.
        normalize (bool): If True, return the relative frequencies of the labels. Default is False.

    Returns:
        dict: A dictionary with label counts or relative frequencies (if normalize=True), sorted by label.
    """
    label_counts = Counter()

    total_samples = 0  # To calculate relative frequencies if normalize=True

    # Iterate through the DataLoader
    for batch in data_loader:
        _, labels = batch  # We only care about the labels (input_ids, labels)

        # Update the Counter with the labels
        label_counts.update(labels.tolist())
        total_samples += labels.size(0)  # Update total sample count

    # If normalize=True, convert counts to relative frequencies
    if normalize:
        label_counts = {
            label: count / total_samples for label, count in label_counts.items()
        }

    # Sort by label (for consistency with pd.value_counts())
    label_counts = dict(sorted(label_counts.items()))

    return label_counts
