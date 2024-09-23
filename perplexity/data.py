# data.py

import datasets
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset
import torch
from torch import Tensor
from typing import Tuple

def setup_dataset(dataset_name: str, seq_max_length: int = 512):
    """Load the wikitext dataset and encode it using the GPT2 tokenizer.

    Args:
        dataset_name (str): "wikitext-2-v1" for small, "wikitext-103-v1" for large.
        seq_max_length (int): Maximum sequence length for the tokenizer.
    
    Returns:
        dataset (datasets.DatasetDict): The encoded wikitext dataset with 'train', 'validation', 'test' splits.
        tokenizer (transformers.GPT2Tokenizer): The GPT2 tokenizer.
    """
    # Load the wikitext dataset
    dataset = datasets.load_dataset("wikitext", dataset_name)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the pad token to the EOS token if it's not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Function to encode examples using the tokenizer
    def encode(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_max_length,
        )

    # Encode the dataset
    dataset = dataset.map(encode, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids"])

    return dataset, tokenizer


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into `bsz` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape `[N]`
        bsz: int, batch size

    Returns:
        Tensor of shape `[N // bsz, bsz]`
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source: Tensor, i: int, bptt: int = 35) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape `[full_seq_len, batch_size]`
        i: int
        bptt: int, sequence length

    Returns:
        tuple (data, target), where data has shape `[bptt, batch_size]` and
        target has shape `[bptt * batch_size]`
    """
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def get_data_tensors(dataset: datasets.DatasetDict, data_fraction: float, batch_size: int) -> Tuple[Tensor, Tensor]:
    """Create batchified train and validation tensors for a subset of the dataset.

    Args:
        dataset (datasets.DatasetDict): The dataset with 'train' and 'validation' splits.
        data_fraction (float): Fraction of the training dataset to use.
        batch_size (int): Batch size for training.

    Returns:
        Tuple containing:
            - train_data (Tensor): Batchified training data tensor.
            - val_data (Tensor): Batchified validation data tensor.
    """
    # Select a subset of the training data
    train_size = int(len(dataset["train"]) * data_fraction)
    train_subset = dataset["train"].select(range(train_size))
    
    # Concatenate all input_ids into a single tensor
    train_data = torch.cat([example["input_ids"] for example in train_subset], dim=0)
    val_data = torch.cat([example["input_ids"] for example in dataset["validation"]], dim=0)

    # Batchify the data
    train_data = batchify(train_data, batch_size)  # Shape: [seq_len, batch_size]
    val_data = batchify(val_data, batch_size // 2)  # You can adjust eval batch size as needed

    return train_data, val_data
