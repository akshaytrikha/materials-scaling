import datasets
from transformers import GPT2Tokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Subset
import os

SEQ_LENGTH = 128


def encode(examples, tokenizer):
    """
    Encode text examples using a specified tokenizer (GPT2Tokenizer or custom BPE tokenizer).

    Args:
        examples (dict): A batch of examples from the dataset.
        tokenizer (PreTrainedTokenizerFast or GPT2Tokenizer): The tokenizer to use for encoding.

    Returns:
        dict: A dictionary containing encoded input_ids, labels, label, and src_key_padding_mask.
    """
    encoded_examples = {
        "input_ids": [],
        "labels": [],
        "label": [],
        "src_key_padding_mask": [],
    }

    # Encode all texts at once
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=SEQ_LENGTH + 1,  # Add 1 for the EOS token
        return_tensors=None,  # Ensure output is lists, not tensors
    )

    # Process the encoded tokens
    encoded_examples["input_ids"] = [ids[:-1] for ids in tokens["input_ids"]]
    encoded_examples["labels"] = [ids[1:] for ids in tokens["input_ids"]]
    encoded_examples["label"] = [[ids[-1]] for ids in tokens["input_ids"]]

    # Create src_key_padding_mask (True for pad tokens, False for others)
    pad_token_id = tokenizer.pad_token_id
    encoded_examples["src_key_padding_mask"] = [
        [token == pad_token_id for token in ids[:-1]] for ids in tokens["input_ids"]
    ]

    return encoded_examples


def setup_dataset(dataset_name: str, tokenizer_name: str):
    """Load WikiText dataset and encode it using the specified tokenizer.

    Args:
        dataset_name (str): Name of the WikiText dataset (e.g., "wikitext-2-raw-v1" or "wikitext-103-raw-v1").
        tokenizer_name (str): Either "gpt2" or "bpe".

    Returns:
        dataset (datasets.DatasetDict): The encoded WikiText dataset.
        tokenizer (PreTrainedTokenizerFast or GPT2Tokenizer): The configured tokenizer.
    """
    # Load the WikiText dataset
    dataset = datasets.load_dataset("wikitext", dataset_name)

    # Initialize the tokenizer based on the specified name
    if tokenizer_name.lower() == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif tokenizer_name.lower() in ["bpe", "bpe_tokenizer"]:
        tokenizer = PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer")
    else:
        raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")

    # Ensure the pad token is set
    if tokenizer.pad_token is None:
        if "<pad>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<pad>"
            print("Set pad_token to '<pad>'.")
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        else:
            # Add a new pad token if none exists
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print("Added and set pad_token to '[PAD]'.")

    # Ensure other special tokens are present
    special_tokens = ["<s>", "</s>", "<unk>", "<mask>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Filter out empty inputs
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Encode the dataset
    dataset = dataset.map(
        encode,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )

    # Set the format of the dataset to PyTorch tensors
    dataset.set_format(
        type="torch", columns=["input_ids", "labels", "label", "src_key_padding_mask"]
    )

    return dataset, tokenizer


def get_dataloaders(
    dataset: datasets.DatasetDict, data_fraction: float, batch_size: int
):
    """Create train and validation dataloaders for a subset of the dataset.

    Args:
        dataset (datasets.DatasetDict): The dataset to create dataloaders from.
        data_fraction (float): Fraction of the dataset to use for training.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        tuple:
            - train_loader (torch.utils.data.DataLoader): Dataloader for the training subset.
            - val_loader (torch.utils.data.DataLoader): Dataloader for the validation subset.
    """
    # Determine the number of training samples based on the data fraction
    train_size = int(len(dataset["train"]) * data_fraction)
    train_subset = Subset(dataset["train"], indices=range(train_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def count_train_tokens(train_loader):
    """Count the total number of tokens in the training data."""
    total_tokens = 0
    for batch in train_loader:
        total_tokens += batch["input_ids"].numel()  # Sum the total number of tokens
    return total_tokens


# if __name__ == "__main__":
#     dataset_name = (
#         "wikitext-103-raw-v1"  # Choose "wikitext-2-raw-v1" or "wikitext-103-raw-v1"
#     )
#     batch_size = 8
#     data_fractions = [0.01, 0.1, 0.25, 0.5, 0.75, 1]

#     # Ensure the tokenizer directory exists
#     if not os.path.exists("bpe_tokenizer"):
#         raise FileNotFoundError(
#             "The 'bpe_tokenizer' directory does not exist. Please run 'tokenizer.py' first."
#         )

#     # Setup dataset and tokenizer
#     dataset, tokenizer = setup_dataset(dataset_name, "bpe")

#     # Iterate over each data fraction and count tokens
#     for fraction in data_fractions:
#         train_loader, _ = get_dataloaders(
#             dataset, data_fraction=fraction, batch_size=batch_size
#         )
#         total_tokens = count_train_tokens(train_loader)
#         print(f"Data Fraction: {fraction*100:.0f}%, Total Tokens: {total_tokens}")
