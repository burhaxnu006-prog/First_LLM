import random


class Embedding:

    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Create random vectors for every token
        self.weights = []

        for _ in range(vocab_size):
            vector = [
                random.uniform(-1, 1)
                for _ in range(embedding_size)
            ]

            self.weights.append(vector)

    def get_embedding(self, token_id):
        return self.weights[token_id]


# Our vocabulary has 7 tokens
vocab_size = 10

# We'll use a tiny 4-dimensional embedding
embedding_size = 4

embedding = Embedding(vocab_size, embedding_size)


# Get the embedding for "Python"
python_id = 3

vector = embedding.get_embedding(python_id)

print("Token ID:")
print(python_id)

print("\nEmbedding:")
print(vector)
