# External
import pickle as pkl
import os
from collections import defaultdict, Counter
import math
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

# Internal
from data import setup_dataset


class NGramModel:
    def __init__(self, n, tokenizer):
        self.n = n
        self.tokenizer = tokenizer
        self.ngram_counts = [defaultdict(Counter) for _ in range(n)]
        self.context_counts = [defaultdict(int) for _ in range(n)]
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
                if int(token.cpu().numpy()) not in special_tokens
            ]
            self.vocab.update(tokens)
            for i in range(self.n, len(tokens)):
                for j in range(1, self.n + 1):
                    context = tuple(tokens[i - j : i])
                    word = tokens[i]
                    self.ngram_counts[j - 1][context][word] += 1
                    self.context_counts[j - 1][context] += 1

    def get_prob(self, context, word, order=None, smoothing=0):
        if order is None:
            order = self.n - 1

        if order < 0:
            return 1 / len(self.vocab)  # Uniform distribution as fallback

        count = self.ngram_counts[order][context][word] + smoothing
        total = self.context_counts[order][context] + smoothing * len(self.vocab)
        if total == 0:
            return self.get_prob(context[1:], word, order - 1, smoothing)
        prob = count / total

        # Use backoff if probability is zero
        if prob == 0:
            return self.get_prob(context[1:], word, order - 1, smoothing)
        return prob

    def perplexity(self, dataset):
        log_prob_sum = 0.0
        total_words = 0

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
                if int(token.cpu().numpy()) not in special_tokens
            ]

            for i in range(len(tokens)):
                context = tuple(tokens[max(0, i - self.n + 1) : i])
                word = tokens[i]
                prob = self.get_prob(context, word, order=len(context), smoothing=0.1)
                log_prob_sum += math.log(prob)
                total_words += 1

        avg_log_prob = log_prob_sum / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as f:
            return pkl.load(f)

    def infer(self, user_input):
        """Infer the probabilities of the next word given a user input."""
        tokens = self.tokenizer.encode(user_input, add_special_tokens=False)

        probabilities = []
        max_prob = -1
        max_prob_word = None

        context = tuple(tokens[max(0, len(tokens) - self.n) :])
        for word in self.vocab:
            prob = self.get_prob(context, word, order=len(context) - 1)
            probabilities.append((context, word, prob))

            if prob > max_prob:
                max_prob = prob
                max_prob_word = word
        return probabilities, self.tokenizer.decode(max_prob_word)


def train(n, train_dataset, val_dataset, tokenizer):
    model_filepath = f"ngram_models/{n}gram_model.pkl"

    # Train n-gram model
    ngram_model = NGramModel(n=n, tokenizer=tokenizer)
    ngram_model.train(train_dataset)
    print(f"{n}-gram Vocabulary Size: {len(ngram_model.vocab)}")
    print(
        f"{n}-gram Unique Contexts: {[len(contexts) for contexts in ngram_model.ngram_counts]}"
    )

    ngram_ppl = ngram_model.perplexity(val_dataset)
    print(f"{n}-gram Perplexity: {ngram_ppl}")

    # Save the model
    ngram_model.save(model_filepath)
    print(f"{n}-gram model saved to {model_filepath}")

    return ngram_model


def inference(n, user_input, ngram_model=None):
    if ngram_model is None:
        model_filepath = f"ngram_models/{n}gram_model.pkl"

        # Load model if it exists
        if os.path.exists(model_filepath):
            ngram_model = NGramModel.load(model_filepath)
        else:
            print(f"{n}-gram model does not exist at {model_filepath}")

    for i in range(10):
        probabilities, max_prob_word = ngram_model.infer(user_input)

        user_input += max_prob_word
        print(user_input)

    return probabilities


def plot_top_k_probs(probabilities, k, tokenizer, title):
    """
    Plot the top-k probabilities as a bar chart.

    Args:
        probabilities: Tensor of shape [vocab_size], the probabilities for the current step.
        k: int, the number of top probabilities to display.
        tokenizer: The tokenizer instance to convert token IDs to readable tokens.
    """
    # Get the top-k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probabilities, k)

    # Convert token indices to actual tokens
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices.tolist()]

    # Plot the probabilities as a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(top_k_tokens, top_k_probs.cpu().numpy())
    plt.title(title)
    plt.xlabel("Tokens")
    plt.ylabel("Probabilities")
    plt.show()


if __name__ == "__main__":
    dataset_name = "wikitext-2-raw-v1"  # or "wikitext-103-v1"

    dataset, tokenizer = setup_dataset(dataset_name)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Create directory for saving models
    os.makedirs("ngram_models", exist_ok=True)

    for n in range(1, 6):
        ngram_model = train(n, train_dataset, val_dataset, tokenizer)

        probabilities = inference(n, "i am a student at")

        # Extract probabilities and convert to tensor
        probabilities = torch.tensor([prob for _, _, prob in probabilities])
        k = 10
        title = f"Top {k} Probability Distribution for {n}-gram"
        plot_top_k_probs(probabilities, k, tokenizer, title)
