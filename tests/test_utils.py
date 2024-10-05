import pytest
import torch
from unittest.mock import patch
from src.utils import generate_sequence

@pytest.fixture
def setup_model_and_data():
    # Create a mock model (you can also use a small real model if you prefer)
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), x.size(1), 100)  # Example: vocab size of 100
    
    model = MockModel()
    idx = torch.randint(0, 100, (2, 10))  # Batch size 2, 10 tokens as input context
    max_new_tokens = 5
    context_size = 10  # Maximum context size to consider
    
    device = torch.device("cpu")
    
    return model, idx, max_new_tokens, context_size, device

@patch('src.generate.get_logits')
def test_generate_sequence_shape(mock_get_logits, setup_model_and_data):
    model, idx, max_new_tokens, context_size, device = setup_model_and_data
    
    # Mock the output of get_logits to simulate next token prediction
    mock_get_logits.return_value = torch.randn(idx.size(0), 100)  # (batch_size, vocab_size)
    
    # Call the function
    generated_sequence = generate_sequence(model, idx, max_new_tokens, context_size)
    
    # Check that the sequence has the expected shape: (B, T + max_new_tokens)
    assert generated_sequence.shape == (idx.size(0), idx.size(1) + max_new_tokens)

def test_generate_no_new_tokens(setup_model_and_data):
    model, idx, _, context_size, device = setup_model_and_data
    max_new_tokens = 0  # No new tokens to generate
    
    # Call the function
    generated_sequence = generate_sequence(model, idx, max_new_tokens, context_size)
    
    # The output should be the same as the input, since no tokens are generated
    assert torch.equal(generated_sequence, idx)

@patch('src.generate.get_logits')
def test_generate_with_limited_context(mock_get_logits, setup_model_and_data):
    model, idx, max_new_tokens, _, device = setup_model_and_data
    limited_context_size = 5  # Use only the last 5 tokens as context

    # Mock the output of get_logits to simulate next token prediction
    mock_get_logits.return_value = torch.randn(idx.size(0), 100)
    
    # Call the function
    generated_sequence = generate_sequence(model, idx, max_new_tokens, limited_context_size)
    
    # Check the shape of the output
    assert generated_sequence.shape == (idx.size(0), idx.size(1) + max_new_tokens)
    
    # Verify that get_logits was called with the correct context size
    for call in mock_get_logits.call_args_list:
        input_idx = call[0][1]
        assert input_idx.size(1) <= limited_context_size  # Context size should be <= limited_context_size

def test_empty_input():
    model = torch.nn.Linear(10, 100)  # Use a simple model for this test
    idx = torch.empty(0, 10)  # Empty batch (0 batch size, 10 tokens)
    max_new_tokens = 3
    context_size = 10
    
    # Check that calling the function raises a ValueError
    with pytest.raises(ValueError, match="Input batch size cannot be zero."):
        generate_sequence(model, idx, max_new_tokens, context_size)

@patch('src.generate.get_logits')
def test_device_handling(mock_get_logits, setup_model_and_data):
    model, idx, max_new_tokens, context_size, device = setup_model_and_data

    # Move model and idx to device
    model.to(device)
    idx = idx.to(device)
    
    # Mock the output of get_logits to simulate next token prediction
    mock_get_logits.return_value = torch.randn(idx.size(0), 100).to(device)
    
    # Call the function
    generated_sequence = generate_sequence(model, idx, max_new_tokens, context_size)
    
    # Ensure the output is on the correct device
    assert generated_sequence.device == device
