import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.train import calc_loss_batch, calc_loss_loader  # Adjust to your actual module

# Mock model class for testing
class MockModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def setup_data():
    # Create mock input and target batches
    input_batch = torch.randn(8, 10, 50)  # Batch of size 8, sequence length 10, hidden dim 50
    target_batch = torch.randint(0, 100, (8, 10))  # Batch of size 8, sequence length 10, vocab size 100
    
    # Create a simple model
    model = MockModel(vocab_size=100, hidden_dim=50)
    
    # Specify device
    device = torch.device("cpu")  # For testing, you can change this to torch.device("cuda") if available
    
    return input_batch, target_batch, model, device

def test_loss_computation(setup_data):
    input_batch, target_batch, model, device = setup_data
    
    # Call the function
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    
    # Check if loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    
def test_device_handling(setup_data):
    input_batch, target_batch, model, device = setup_data
    
    # Call the function
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    
    # Ensure model and input tensors are on the right device
    assert input_batch.device == device
    assert target_batch.device == device
    assert next(model.parameters()).device == device

def test_empty_input():
    model = MockModel(vocab_size=100, hidden_dim=50)
    device = torch.device("cpu")
    
    # Empty input and target batch
    input_batch = torch.randn(0, 10, 50)  # Empty batch
    target_batch = torch.randint(0, 100, (0, 10))  # Empty batch
    
    # Call the function, expect no errors
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    
    # Loss should still be a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

def test_mismatched_shapes():
    model = MockModel(vocab_size=100, hidden_dim=50)
    device = torch.device("cpu")
    
    # Mismatched input and target batch
    input_batch = torch.randn(8, 10, 50)  # Batch size 8, sequence length 10
    target_batch = torch.randint(0, 100, (8, 9))  # Wrong shape: sequence length should be 10

    # Expecting a ValueError due to mismatched shapes
    with pytest.raises(ValueError, match="Expected input batch_size.*to match target batch_size.*"):
        calc_loss_batch(input_batch, target_batch, model, device)


@pytest.fixture
def setup_dataloader():
    # Create a simple dataset and dataloader
    input_data = torch.randn(64, 10, 50)  # 64 samples, seq length 10, hidden dim 50
    target_data = torch.randint(0, 100, (64, 10))  # 64 samples, seq length 10, vocab size 100
    
    dataset = TensorDataset(input_data, target_data)
    data_loader = DataLoader(dataset, batch_size=8)  # Batch size of 8
    
    # Create a simple model
    model = MockModel(vocab_size=100, hidden_dim=50)
    
    # Specify device (CPU for this test, can be GPU if available)
    device = torch.device("cpu")
    
    return data_loader, model, device

def test_average_loss_computation(setup_dataloader):
    data_loader, model, device = setup_dataloader
    
    # Call the function
    avg_loss = calc_loss_loader(data_loader, model, device)
    
    # Ensure the loss is a float
    assert isinstance(avg_loss, float)
    assert avg_loss > 0  # Loss should be positive

def test_empty_data_loader():
    model = MockModel(vocab_size=100, hidden_dim=50)
    device = torch.device("cpu")
    
    # Create an empty data loader
    empty_dataset = TensorDataset(torch.empty(0, 10, 50), torch.empty(0, 10))
    empty_loader = DataLoader(empty_dataset, batch_size=8)
    
    # Call the function
    avg_loss = calc_loss_loader(empty_loader, model, device)
    
    # Ensure NaN is returned for empty data loader
    assert torch.isnan(torch.tensor(avg_loss)), "Expected NaN for empty data loader"

def test_num_batches_limit(setup_dataloader):
    data_loader, model, device = setup_dataloader
    
    # Limit the number of batches to 2
    avg_loss = calc_loss_loader(data_loader, model, device, num_batches=2)
    
    # Ensure the loss is a float and greater than 0
    assert isinstance(avg_loss, float)
    assert avg_loss > 0  # Loss should be positive
    
    # Check that only 2 batches were processed (by ensuring average loss is reasonable)
    total_batches = len(data_loader)
    assert total_batches >= 2, "Test dataset should have more than 2 batches"

def test_device_handling(setup_dataloader):
    data_loader, model, device = setup_dataloader
    
    # Call the function
    avg_loss = calc_loss_loader(data_loader, model, device)
    
    # Ensure the model and data are on the right device
    assert next(model.parameters()).device == device
    for input_batch, target_batch in data_loader:
        assert input_batch.device == device
        assert target_batch.device == device