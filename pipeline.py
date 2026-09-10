import math
import random
from transformer import TransformerBlock


# -------------------------
# TOKENIZER
# -------------------------

vocab = {
    "I": 0,
    "am": 1,
    "learning": 2,
    "Python": 3,
    ".": 4,
    "is": 5,
    "powerful": 6
}

text = "I am learning Python."

words = text.replace(".", " .").split()

token_ids = []

for word in words:
    token_ids.append(vocab[word])


print("Text:")
print(text)

print("\nToken IDs:")
print(token_ids)


# -------------------------
# TOKEN EMBEDDING
# -------------------------

vocab_size = len(vocab)
embedding_size = 4

embedding_matrix = []

for token in range(vocab_size):

    vector = []

    for dimension in range(embedding_size):

        vector.append(
            random.uniform(-1, 1)
        )

    embedding_matrix.append(vector)


token_vectors = []

for token_id in token_ids:

    vector = embedding_matrix[token_id]

    token_vectors.append(vector)


print("\nToken Embeddings:")

for i, vector in enumerate(token_vectors):

    print(
        "Token",
        i + 1,
        "ID",
        token_ids[i],
        "→",
        vector
    )


# -------------------------
# POSITIONAL EMBEDDING
# -------------------------

max_length = 10

position_vectors = []

for position in range(max_length):

    vector = []

    for dimension in range(embedding_size):

        value = position / max_length

        vector.append(value)

    position_vectors.append(vector)


# -------------------------
# ADD TOKEN + POSITION
# -------------------------

final_vectors = []

for i in range(len(token_vectors)):

    token_vector = token_vectors[i]

    position_vector = position_vectors[i]

    combined = []

    for dimension in range(embedding_size):

        value = (
            token_vector[dimension]
            + position_vector[dimension]
        )

        combined.append(value)

    final_vectors.append(combined)


print("\nPosition Embeddings:")

for i in range(len(token_vectors)):

    print(
        "Position",
        i,
        "→",
        position_vectors[i]
    )


print("\nToken + Position:")

for i, vector in enumerate(final_vectors):

    print(
        "Token",
        i + 1,
        "→",
        vector
    )


# -------------------------
# TRANSFORMER
# -------------------------

print("\n==============================")
print("Sending vectors to Transformer")
print("==============================")

transformer = TransformerBlock(
    embedding_size=4,
    hidden_size=8
)

final_outputs = transformer.forward(final_vectors)

# -------------------------
# NEXT-TOKEN PREDICTION
# -------------------------

print("\n==============================")
print("Next-Token Prediction")
print("==============================")


# Get the final Transformer vector
last_vector = transformer.forward(final_vectors)[-1]

print("\nLast Transformer Vector:")
print(last_vector)


# Output matrix
# 4 input dimensions → 7 vocabulary tokens

output_matrix = []

for row in range(embedding_size):

    values = []

    for column in range(vocab_size):

        values.append(
            random.uniform(-1, 1)
        )

    output_matrix.append(values)


# Calculate logits

logits = []

for token in range(vocab_size):

    score = 0

    for dimension in range(embedding_size):

        score += (
            last_vector[dimension]
            * output_matrix[dimension][token]
        )

    logits.append(score)


print("\nLogits:")

for token_id, score in enumerate(logits):

    print(
        vocab,
        "ID",
        token_id,
        "→",
        score
    )


# -------------------------
# SOFTMAX
# -------------------------

exp_scores = []

for score in logits:

    exp_scores.append(
        math.exp(score)
    )


total = sum(exp_scores)

probabilities = []

for value in exp_scores:

    probabilities.append(
        value / total
    )


print("\nProbabilities:")

for token_id, probability in enumerate(probabilities):

    print(
        "Token ID",
        token_id,
        "→",
        probability
    )


# -------------------------
# FIND HIGHEST PROBABILITY
# -------------------------

best_token_id = probabilities.index(
    max(probabilities)
)


print("\nPredicted Next Token:")

for word, token_id in vocab.items():

    if token_id == best_token_id:

        print(word)
