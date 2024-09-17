import datasets
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset


def setup_dataset(dataset_name: str, seq_max_length: int = 512):
    """Load the wikitext dataset and encode it using the GPT2 tokenizer.

    Args:
        dataset_name (str): small is "wikitext-2-v1", large is "wikitext-103-v1" which is 50x bigger
        seq_max_length (int): Maximum sequence length for the tokenizer
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

    # Function to encode examples using the tokenizer
    def encode(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_max_length,
        )

    # Encode the dataset
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])

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
    val_size = int(len(dataset["validation"]) * data_fraction)

    train_subset = Subset(dataset["train"], indices=range(train_size))
    val_subset = Subset(dataset["validation"], indices=range(val_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
