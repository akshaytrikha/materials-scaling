import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Load the WikiText dataset (choose 'wikitext-2-raw-v1' or 'wikitext-103-raw-v1')
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Combine all text data into a single list
texts = []
for split in ["train", "validation", "test"]:
    for item in dataset[split]:
        texts.append(item["text"])

# Optionally, save the combined text to a single file
with open("wikitext_combined.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

# Initialize the tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on the combined WikiText file
tokenizer.train(
    files=["wikitext_combined.txt"],
    vocab_size=10000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

# Save the tokenizer using the `save` method to include tokenizer.json
os.makedirs("bpe_tokenizer", exist_ok=True)
tokenizer.save("bpe_tokenizer/tokenizer.json")
tokenizer.save_model("bpe_tokenizer")
