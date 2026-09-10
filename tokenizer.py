import re


class Tokenizer:

    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    def build_vocab(self, text):
        tokens = self.tokenize(text)

        for token in tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)

                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token

    def encode(self, text):
        tokens = self.tokenize(text)

        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids):
        return [self.reverse_vocab[token_id] for token_id in token_ids]


# Training text
text = "This is just an prototype of tokenizer,to show how an sentence get's tokenizied.."

# Create tokenizer
tokenizer = Tokenizer()

# Build vocabulary
tokenizer.build_vocab(text)

print("Vocabulary:")
print(tokenizer.vocab)

encoded = tokenizer.encode(text)

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)
