# External
from transformers import GPT2Tokenizer
from collections import defaultdict, Counter
import math
from tqdm.auto import tqdm

# Internal
from data import setup_dataset


from collections import defaultdict, Counter
import math
from tqdm.auto import tqdm


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
            tokens = [token for token in tokens if token not in special_tokens]
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
            tokens = [token for token in tokens if token not in special_tokens]

            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                word = tokens[i + self.n - 1]
                prob = self.get_prob(context, word)
                log_prob += math.log(prob)  # Natural logarithm
                N += 1

        return math.exp(-log_prob / N)  # Natural exponent


dataset_name = "wikitext-2-v1"  # or "wikitext-103-v1"

dataset, tokenizer = setup_dataset(dataset_name)
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# train n-gram models
for n in range(2, 6):
    ngram_model = NGramModel(n=n, tokenizer=tokenizer)
    ngram_model.train(train_dataset)
    print(f"{n}-gram Vocabulary Size: {len(ngram_model.vocab)}")
    print(f"{n}-gram Unique Contexts: {len(ngram_model.ngram_counts)}")

    ngram_ppl = ngram_model.perplexity(val_dataset)
    print(f"{n}-gram Perplexity: {ngram_ppl}\n")
