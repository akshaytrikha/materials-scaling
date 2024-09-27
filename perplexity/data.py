import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import GPT2Tokenizer, AdamW
import datasets
from typing import List, Tuple
import numpy as np

class LanguageModelingDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        sequence_length: int = 32,
        stride: int = 1,
        max_samples: int = None
    ):
        """
        Initializes the dataset by tokenizing the texts and creating input-target pairs.

        Args:
            texts (List[str]): List of raw text data.
            tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
            sequence_length (int): Length of each input sequence.
            stride (int): Number of tokens to move the window for the next sequence.
            max_samples (int, optional): Maximum number of sequences to include. Useful for debugging.
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.stride = stride
        self.input_ids = []
        self.labels = []

        for text in texts:
            # Tokenize the text without adding special tokens, padding, or truncation
            tokens = tokenizer.encode(
                text,
                add_special_tokens=False
            )

            token_length = len(tokens)

            # Skip texts that are too short to create at least one sequence
            if token_length < sequence_length + 1:
                continue

            # Sliding window to create input-target pairs
            for i in range(0, token_length - sequence_length, stride):
                input_chunk = tokens[i:i + sequence_length]
                target_chunk = tokens[i + 1:i + sequence_length + 1]

                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.labels.append(torch.tensor(target_chunk, dtype=torch.long))

                if max_samples and len(self.input_ids) >= max_samples:
                    break
            if max_samples and len(self.input_ids) >= max_samples:
                break

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

def setup_tokenizer() -> GPT2Tokenizer:
    """
    Initializes and configures the GPT-2 tokenizer.

    Returns:
        GPT2Tokenizer: Configured GPT-2 tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer

def load_texts(dataset_name: str) -> Tuple[List[str], List[str]]:
    """
    Loads texts from the specified Wikitext dataset.

    Args:
        dataset_name (str): "wikitext-2-v1" or "wikitext-103-v1"

    Returns:
        Tuple[List[str], List[str]]: Training and validation texts.
    """
    raw_datasets = datasets.load_dataset("wikitext", dataset_name)
    train_texts = raw_datasets['train']['text']
    val_texts = raw_datasets['validation']['text']
    return train_texts, val_texts

def create_datasets(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer: GPT2Tokenizer,
    sequence_length: int,
    stride: int,
) -> Tuple[LanguageModelingDataset, LanguageModelingDataset]:
    """
    Creates training and validation datasets.

    Args:
        train_texts (List[str]): Training texts.
        val_texts (List[str]): Validation texts.
        tokenizer (GPT2Tokenizer): GPT-2 tokenizer.
        sequence_length (int): Length of each input sequence.
        stride (int): Stride for the sliding window.
        max_train_samples (int, optional): Max training samples.
        max_val_samples (int, optional): Max validation samples.

    Returns:
        Tuple[LanguageModelingDataset, LanguageModelingDataset]: Training and validation datasets.
    """
    train_dataset = LanguageModelingDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    val_dataset = LanguageModelingDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    return train_dataset, val_dataset

def get_dataloaders(
    train_dataset: LanguageModelingDataset,
    val_dataset: LanguageModelingDataset,
    batch_size: int,
    train_fraction: float = 1.0,
    val_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation datasets with specified fractions.

    Args:
        train_dataset (LanguageModelingDataset): Full training dataset.
        val_dataset (LanguageModelingDataset): Full validation dataset.
        batch_size (int): Batch size.
        train_fraction (float, optional): Fraction of the training dataset to use (0.0 - 1.0). Defaults to 1.0.
        val_fraction (float, optional): Fraction of the validation dataset to use (0.0 - 1.0). Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # # Set random seed for reproducibility
    # random_seed = 42
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(random_seed)

    # Calculate subset sizes
    train_subset_size = int(len(train_dataset) * train_fraction)
    val_subset_size = int(len(val_dataset) * val_fraction)

    # Generate random subset indices
    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False).tolist() if train_fraction < 1.0 else list(range(len(train_dataset)))
    val_indices = np.random.choice(len(val_dataset), val_subset_size, replace=False).tolist() if val_fraction < 1.0 else list(range(len(val_dataset)))

    # Create Subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"Training subset size: {len(train_subset)} (Fraction: {train_fraction})")
    print(f"Validation subset size: {len(val_subset)} (Fraction: {val_fraction})")

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,      # Shuffle for better training
        drop_last=True     # Ensures consistent batch sizes
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader


# def train_model(model: GPT2LMHeadModel, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 1, lr: float = 5e-5):
    """
    Trains the GPT-2 model.

    Args:
        model (GPT2LMHeadModel): GPT-2 model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

def main():
    # Configuration
    dataset_name = "wikitext-2-v1"  # or "wikitext-103-v1" for a larger dataset
    sequence_length = 32             # Length of each input sequence
    stride = 1                       # Stride for the sliding window
    batch_size = 8                   # Batch size

    # Setup tokenizer
    tokenizer = setup_tokenizer()

    # Load texts
    train_texts, val_texts = load_texts(dataset_name)

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=stride,
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader, val_loader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
    )

    # Fetch a single batch from the training DataLoader for demonstration
    batch = next(iter(train_loader))

    print(f"\nBatch 'input_ids' shape: {batch['input_ids'].shape}")  # Should be [batch_size, sequence_length]
    print(f"First sequence in 'input_ids': {batch['input_ids'][0]}")  # Token IDs

    input_ids = batch['input_ids']
    labels = batch['labels']

    # Decode the first 3 sequences in the batch for demonstration
    num_samples_to_display = 3
    for i in range(num_samples_to_display):
        input = input_ids[i]
        label = labels[i]

        print(f"\nSample {i + 1}:")
        print("Input:")
        print(input)
        print("\nLabel:")
        print(label)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
