import datasets
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset

SEQ_LENGTH = 128


# Function to encode examples using the tokenizer
def encode(examples, tokenizer, seq_length):
    """Encode text examples using GPT2 tokenizer"""

    encoded_examples = {
        "input_ids": [],
        "labels": [],
        "label": [],
        "src_key_padding_mask": [],
    }
    for text in examples["text"]:
        if text.strip():  # Check if the text is not empty or just whitespace
            tokens = tokenizer.encode(text)

            if len(tokens) < seq_length:
                # Pad from the left
                padding = [tokenizer.pad_token_id] * (seq_length - len(tokens))
                tokens = padding + tokens
            elif len(tokens) > seq_length:
                # Truncate to seq_length
                tokens = tokens[:seq_length]

            encoded_examples["input_ids"].append(tokens[:-1])
            encoded_examples["labels"].append(tokens[1:])
            encoded_examples["label"].append([tokens[-1]])

            # Create src_key_padding_mask (True for pad tokens, False for others)
            src_key_padding_mask = [
                token == tokenizer.pad_token_id for token in tokens[:-1]
            ]
            encoded_examples["src_key_padding_mask"].append(src_key_padding_mask)

    if not encoded_examples["input_ids"]:
        # If no valid text was found, return a dictionary with pad tokens
        print("No valid text found")
        encoded_examples["input_ids"].append(
            [tokenizer.pad_token_id] * (seq_length - 1)
        )
        encoded_examples["labels"].append([tokenizer.pad_token_id] * (seq_length - 1))
        encoded_examples["label"].append([tokenizer.pad_token_id])
        encoded_examples["src_key_padding_mask"].append([True] * (seq_length - 1))

    return encoded_examples


def filter_pad_data(example, tokenizer):
    """Filter out examples are all pad tokens"""
    return not all(token == tokenizer.pad_token_id for token in example["input_ids"])


def setup_dataset(dataset_name: str):
    """Load wikitext dataset and encode it using the GPT2 tokenizer.

    Args:
        dataset_name (str): small is "wikitext-2-v1", large is "wikitext-103-v1" which is 50x bigger
    Returns:
        dataset (datasets.Dataset): The encoded wikitext dataset
        tokenizer (transformers.GPT2Tokenizer): The GPT2 tokenizer
    """
    # Load the wikitext dataset
    dataset = datasets.load_dataset("wikitext", dataset_name)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the pad token to the EOS token if it's not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode the dataset
    dataset = dataset.map(
        encode,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "seq_length": SEQ_LENGTH},
        remove_columns=dataset["train"].column_names,
    )

    dataset = dataset.filter(filter_pad_data, fn_kwargs={"tokenizer": tokenizer})
    dataset.set_format(
        type="torch", columns=["input_ids", "labels", "label", "src_key_padding_mask"]
    )

    return dataset, tokenizer


def get_dataloaders(dataset: datasets.Dataset, data_fraction: float, batch_size: int):
    """Create train and validation dataloaders for a subset of the dataset.

    Args:
        dataset (datasets.Dataset): The dataset to create dataloaders from
        data_fraction (float): Fraction of the dataset to use
        batch_size (int): Batch size for the dataloaders
    Returns:
        train_loader (torch.utils.data.DataLoader): Dataloader for the training subset
        val_loader (torch.utils.data.DataLoader): Dataloader for the validation subset
    """
    # Create a subset of the dataset
    train_size = int(len(dataset["train"]) * data_fraction)

    train_subset = Subset(dataset["train"], indices=range(train_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
