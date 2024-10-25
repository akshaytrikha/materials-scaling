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
    # Encode all texts at once
    tokens = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=seq_length
    )

    # Process the encoded tokens
    encoded_examples["input_ids"] = [ids[:-1] for ids in tokens["input_ids"]]
    encoded_examples["labels"] = [ids[1:] for ids in tokens["input_ids"]]
    encoded_examples["label"] = [[ids[-1]] for ids in tokens["input_ids"]]

    # Create src_key_padding_mask (True for pad tokens, False for others)
    encoded_examples["src_key_padding_mask"] = [
        [token == tokenizer.pad_token_id for token in ids[:-1]]
        for ids in tokens["input_ids"]
    ]

    return encoded_examples


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

    # Filter out empty inputs first
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Encode the dataset
    dataset = dataset.map(
        encode,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "seq_length": SEQ_LENGTH + 1},
        remove_columns=dataset["train"].column_names,
    )

    dataset.set_format(
        type="torch", columns=["input_ids", "labels", "label", "src_key_padding_mask"]
    )

    return dataset, tokenizer


def get_dataloaders(
    dataset: datasets.Dataset, data_fraction: float, batch_size: int, sampler=None
):
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

    shuffle = sampler is None # sampler option is mutually exclusive with shuffle
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=batch_size, shuffle=shuffle, sampler=sampler
    )

    return train_loader, val_loader


def count_train_tokens(train_loader):
    """Count the total number of tokens in the training data."""
    total_tokens = 0
    for batch in train_loader:
        total_tokens += batch["input_ids"].numel()  # Sum the total number of tokens
    return total_tokens


if __name__ == "__main__":
    dataset_name = (
        "wikitext-103-v1"  # You can switch to "wikitext-103-v1" for the larger version
    )
    seq_length = 512
    batch_size = 8
    data_fractions = [0.01, 0.1, 0.25, 0.5, 0.75, 1]

    # Setup dataset and tokenizer
    dataset, tokenizer = setup_dataset(dataset_name, seq_length)

    # Iterate over each data fraction
    for fraction in data_fractions:
        train_loader, _ = get_dataloaders(
            dataset, data_fraction=fraction, batch_size=batch_size
        )
        print(count_train_tokens(train_loader))
