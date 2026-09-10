import math


def dot_product(a, b):
    total = 0

    for i in range(len(a)):
        total += a[i] * b[i]

    return total


def softmax(scores):
    exp_scores = []

    for score in scores:
        exp_scores.append(math.exp(score))

    total = sum(exp_scores)

    return [
        value / total
        for value in exp_scores
    ]


def matrix_vector_multiply(vector, matrix):
    result = []

    for row in matrix:
        value = dot_product(vector, row)
        result.append(value)

    return result


def self_attention(tokens):

    print("Input tokens:")
    print(tokens)

    # --------------------------------
    # 1. Create simple Q, K and V
    # --------------------------------

    query_matrix = [
        [1, 0],
        [0, 1]
    ]

    key_matrix = [
        [1, 0],
        [0, 1]
    ]

    value_matrix = [
        [1, 0],
        [0, 1]
    ]

    # --------------------------------
    # 2. Convert tokens into Q, K, V
    # --------------------------------

    queries = []
    keys = []
    values = []

    for token in tokens:

        q = matrix_vector_multiply(token, query_matrix)
        k = matrix_vector_multiply(token, key_matrix)
        v = matrix_vector_multiply(token, value_matrix)

        queries.append(q)
        keys.append(k)
        values.append(v)

    print("\nQueries:")
    print(queries)

    print("\nKeys:")
    print(keys)

    print("\nValues:")
    print(values)

    # --------------------------------
    # 3. Calculate attention
    #    for every token
    # --------------------------------

    all_outputs = []

    for query in queries:

        scores = []

        for key in keys:

            score = dot_product(query, key)

            # Scale the score
            score = score / math.sqrt(len(key))

            scores.append(score)

        weights = softmax(scores)

        # --------------------------------
        # 4. Weighted sum of values
        # --------------------------------

        output = []

        for dimension in range(len(values[0])):

            total = 0

            for i in range(len(values)):

                total += weights[i] * values[i][dimension]

            output.append(total)

        all_outputs.append(output)

    print("\nAttention outputs:")

    for i, output in enumerate(all_outputs):

        print("Token", i + 1, "→", output)


# --------------------------------
# Example
# --------------------------------

tokens = [
    [1, 0],
    [0, 1],
    [1, 1]
]

self_attention(tokens)
