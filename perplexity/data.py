from datasets import load_dataset
from transformers import GPT2Tokenizer


def setup_dataset(dataset_name: str):
    """Load the wikitext dataset and encode it using the GPT2 tokenizer.

    Args:
        dataset_name (str): small is "wikitext-2-v1", large is "wikitext-103-v1" which is 50x bigger
    Returns:
        dataset (datasets.Dataset): The encoded wikitext dataset
        tokenizer (transformers.GPT2Tokenizer): The GPT2 tokenizer
    """
    # Load the wikitext dataset
    dataset = load_dataset("wikitext", dataset_name)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the pad token to the EOS token if it's not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Function to encode examples using the tokenizer
    def encode(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    # Encode the dataset
    dataset = dataset.map(encode, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])

    return dataset, tokenizer
