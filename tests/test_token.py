import torch
import pytest
import tiktoken
from src.token import text_to_token_ids, token_ids_to_text

# Set up tiktoken tokenizer (using GPT-2 tokenizer as an example)
@pytest.fixture
def tokenizer():
    return tiktoken.get_encoding("gpt2")

# Test the text_to_token_ids function on CPU
def test_text_to_token_ids_cpu(tokenizer):
    text = "Hello, world!"
    
    # Call text_to_token_ids
    result = text_to_token_ids(text, tokenizer)

    # Expected tensor
    expected_encoded = tokenizer.encode(text)
    expected_tensor = torch.tensor(expected_encoded).unsqueeze(0)  # Add batch dimension

    assert torch.equal(result, expected_tensor), "The output tensor does not match the expected result."

# Test the text_to_token_ids function on GPU if available
def test_text_to_token_ids_gpu(tokenizer):
    text = "Hello, world!"
    
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Call text_to_token_ids on GPU
        result = text_to_token_ids(text, tokenizer, device=device)

        # Expected tensor on GPU
        expected_encoded = tokenizer.encode(text)
        expected_tensor = torch.tensor(expected_encoded, device=device).unsqueeze(0)

        assert torch.equal(result, expected_tensor), "The output tensor on GPU does not match the expected result."

# Test the token_ids_to_text function
def test_token_ids_to_text(tokenizer):
    text = "Hello, world!"
    
    # Create tensor (simulating output from model)
    encoded_ids = tokenizer.encode(text)
    token_ids = torch.tensor([encoded_ids])

    # Call token_ids_to_text
    result = token_ids_to_text(token_ids, tokenizer)

    assert result == text, "The decoded text does not match the original input."

# Test both functions in sequence (text_to_token_ids -> token_ids_to_text)
def test_text_to_token_ids_then_token_ids_to_text(tokenizer):
    text = "Hello, world!"
    
    # Step 1: Tokenize the text to tensor
    token_ids = text_to_token_ids(text, tokenizer)

    # Step 2: Decode the tensor back to text
    decoded_text = token_ids_to_text(token_ids, tokenizer)

    assert decoded_text == text, "The decoded text does not match the original input after full tokenization cycle."
