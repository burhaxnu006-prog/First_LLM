import math
import random


# --------------------------------
# Basic vector operations
# --------------------------------

def dot_product(a, b):

    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def add_vectors(a, b):

    return [
        a[i] + b[i]
        for i in range(len(a))
    ]


def relu(vector):

    return [
        max(0, value)
        for value in vector
    ]


def softmax(scores):

    exp_scores = []

    for score in scores:
        exp_scores.append(math.exp(score))

    total = sum(exp_scores)

    return [
        value / total
        for value in exp_scores
    ]


# --------------------------------
# Matrix multiplication
# --------------------------------

def matrix_multiply(vector, matrix):

    result = []

    for column in range(len(matrix[0])):

        total = 0

        for row in range(len(vector)):
            total += vector[row] * matrix[row][column]

        result.append(total)

    return result


# --------------------------------
# Layer Normalization
# --------------------------------

def layer_norm(vector):

    mean = sum(vector) / len(vector)

    variance = 0

    for value in vector:
        variance += (value - mean) ** 2

    variance = variance / len(vector)

    epsilon = 1e-5

    normalized = []

    for value in vector:

        result = (
            (value - mean)
            / math.sqrt(variance + epsilon)
        )

        normalized.append(result)

    return normalized


# --------------------------------
# Transformer Block
# --------------------------------

class TransformerBlock:

    def __init__(self, embedding_size, hidden_size):

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Q, K, V weight matrices

        self.WQ = self.random_matrix(
            embedding_size,
            embedding_size
        )

        self.WK = self.random_matrix(
            embedding_size,
            embedding_size
        )

        self.WV = self.random_matrix(
            embedding_size,
            embedding_size
        )

        # Feed-forward weights

        self.W1 = self.random_matrix(
            embedding_size,
            hidden_size
        )

        self.W2 = self.random_matrix(
            hidden_size,
            embedding_size
        )

    def random_matrix(self, rows, columns):

        matrix = []

        for _ in range(rows):

            row = []

            for _ in range(columns):
                row.append(random.uniform(-1, 1))

            matrix.append(row)

        return matrix

    # --------------------------------
    # Self-Attention
    # --------------------------------

    def attention(self, tokens):

        queries = []
        keys = []
        values = []

        for token in tokens:

            q = matrix_multiply(token, self.WQ)
            k = matrix_multiply(token, self.WK)
            v = matrix_multiply(token, self.WV)

            queries.append(q)
            keys.append(k)
            values.append(v)

        attention_outputs = []

        for query in queries:

            scores = []

            for key in keys:

                score = dot_product(query, key)

                score = score / math.sqrt(
                    len(key)
                )

                scores.append(score)

            weights = softmax(scores)

            output = []

            for dimension in range(
                len(values[0])
            ):

                total = 0

                for i in range(len(values)):

                    total += (
                        weights[i]
                        * values[i][dimension]
                    )

                output.append(total)

            attention_outputs.append(output)

        return attention_outputs

    # --------------------------------
    # Feed-Forward Network
    # --------------------------------

    def feed_forward(self, token):

        hidden = matrix_multiply(
            token,
            self.W1
        )

        hidden = relu(hidden)

        output = matrix_multiply(
            hidden,
            self.W2
        )

        return output

    # --------------------------------
    # Complete Transformer Block
    # --------------------------------

    def forward(self, tokens):

        print("\nInput:")
        print(tokens)

        # 1. Self-Attention

        attention_outputs = self.attention(
            tokens
        )

        print("\nAfter Self-Attention:")
        print(attention_outputs)

        # 2. Residual + LayerNorm

        normalized_1 = []

        for i in range(len(tokens)):

            residual = add_vectors(
                tokens[i],
                attention_outputs[i]
            )

            normalized = layer_norm(
                residual
            )

            normalized_1.append(normalized)

        print("\nAfter Attention + Residual + LayerNorm:")
        print(normalized_1)

        # 3. Feed-Forward

        ff_outputs = []

        for token in normalized_1:

            output = self.feed_forward(token)

            ff_outputs.append(output)

        print("\nAfter Feed-Forward:")
        print(ff_outputs)

        # 4. Second Residual + LayerNorm

        final_outputs = []

        for i in range(len(normalized_1)):

            residual = add_vectors(
                normalized_1[i],
                ff_outputs[i]
            )

            normalized = layer_norm(
                residual
            )

            final_outputs.append(normalized)

        print("\nFinal Transformer Output:")
        print(final_outputs)

        return final_outputs


# --------------------------------
# Example
# --------------------------------
