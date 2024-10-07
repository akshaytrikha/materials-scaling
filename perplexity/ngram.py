# External
import pickle
import os
from transformers import GPT2Tokenizer
from collections import defaultdict, Counter
import math
from tqdm.auto import tqdm

# Internal
from data import setup_dataset


class NGramModel:
    def __init__(self, n, tokenizer):
        self.n = n
        self.tokenizer = tokenizer
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()

    def train(self, dataset):
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        special_tokens = {pad_token_id, bos_token_id, eos_token_id}

        for example in tqdm(dataset, desc="Training"):
            tokens = example["input_ids"]

            # Filter out special tokens
            tokens = [
                int(token.cpu().numpy())
                for token in tokens
                if token not in special_tokens
            ]

            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

    def get_prob(self, context, word, smoothing=1):
        count = self.ngram_counts[context][word] + smoothing
        total = self.context_counts[context] + smoothing * len(self.vocab)
        return count / total

    def perplexity(self, dataset):
        log_prob = 0.0
        N = 0

        special_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id,
        }

        for example in tqdm(dataset, desc="Calculating Perplexity"):
            tokens = example["input_ids"]
            # Filter out special tokens
            tokens = [
                int(token.cpu().numpy())
                for token in tokens
                if token not in special_tokens
            ]

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                prob = self.get_prob(context, word)
                log_prob += math.log(prob)  # Natural logarithm
                N += 1

        return math.exp(-log_prob / N)  # Natural exponent

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def infer_ngram_model(ngram_model, user_input, tokenizer):
    """Infer the probabilities of the next word given a user input using an n-gram model."""

    # Tokenize the user input
    tokens = ngram_model.tokenizer.encode(user_input, add_special_tokens=False)

    # Initialize a list to store probabilities
    probabilities = []

    # Initialize variables to track the word with the highest probability
    max_prob = -1
    max_prob_word = None

    # Loop through the tokens to create n-grams
    for i in range(len(tokens) - ngram_model.n + 1):
        context = tuple(tokens[i : i + ngram_model.n - 1])
        word = tokens[i + ngram_model.n - 1]
        prob = ngram_model.get_prob(context, word)
        probabilities.append((context, word, prob))

        # Update the word with the highest probability
        if prob > max_prob:
            max_prob = prob
            max_prob_word = word

    return probabilities, tokenizer.decode(max_prob_word)


def train(n, train_dataset, val_dataset, tokenizer):
    model_filepath = f"ngram_models/{n}gram_model.pkl"

    # Train n-gram model
    ngram_model = NGramModel(n=n, tokenizer=tokenizer)
    ngram_model.train(train_dataset)
    print(f"{n}-gram Vocabulary Size: {len(ngram_model.vocab)}")
    print(f"{n}-gram Unique Contexts: {len(ngram_model.ngram_counts)}")

    ngram_ppl = ngram_model.perplexity(val_dataset)
    print(f"{n}-gram Perplexity: {ngram_ppl}")

    # Save the model
    ngram_model.save(model_filepath)
    print(f"{n}-gram model saved to {model_filepath}")

    return ngram_model


def inference(n, tokenizer, user_input):
    model_filepath = f"ngram_models/{n}gram_model.pkl"

    # Load model if it exists
    if os.path.exists(model_filepath):
        ngram_model = NGramModel.load(model_filepath)
    else:
        print(f"{n}-gram model does not exist at {model_filepath}")

    for i in range(5):
        probabilities, max_prob_word = infer_ngram_model(
            ngram_model, user_input, tokenizer
        )
        user_input += max_prob_word
        print(user_input)


if __name__ == "__main__":
    dataset_name = "wikitext-2-v1"  # or "wikitext-103-v1"

    dataset, tokenizer = setup_dataset(dataset_name)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Create directory for saving models
    os.makedirs("ngram_models", exist_ok=True)

    for n in range(1, 6):
        train(n, train_dataset, val_dataset, tokenizer)
        inference(n, tokenizer, "Park has a black MacBook that")
