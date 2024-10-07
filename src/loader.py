import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


class SPAMSmsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        """
        Args:
            dataframe (pd.DataFrame): The dataset containing 'Label' and 'Text' columns.
            tokenizer (tiktoken.Encoding): A pre-initialized tiktoken tokenizer.
            max_length (int): The maximum sequence length for tokenization.
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Convert labels to numerical values (spam: 1, ham: 0)
        self.labels = self.data['Label'].map({'ham': 0, 'spam': 1}).values
        self.texts = self.data['Text'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label for the current index
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text using tiktoken
        tokens = self.tokenizer.encode(text)

        # Truncate or pad tokens to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Padding with 0s

        # Convert to PyTorch tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        return (input_ids, torch.tensor(label, dtype=torch.long))
