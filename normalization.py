import math


def add_vectors(a, b):

    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])

    return result


def layer_norm(vector):

    # Calculate mean
    mean = sum(vector) / len(vector)

    # Calculate variance
    variance = 0

    for value in vector:
        variance += (value - mean) ** 2

    variance = variance / len(vector)

    # Small value to prevent division by zero
    epsilon = 1e-5

    # Normalize
    normalized = []

    for value in vector:

        result = (
            (value - mean)
            / math.sqrt(variance + epsilon)
        )

        normalized.append(result)

    return normalized


# --------------------------------
# Example
# --------------------------------

input_vector = [0.8, 0.6]

transformation = [-0.2, -0.13]

print("Input:")
print(input_vector)

print("\nTransformation:")
print(transformation)


# Residual connection
residual = add_vectors(
    input_vector,
    transformation
)

print("\nAfter Residual Connection:")
print(residual)


# Layer normalization
normalized = layer_norm(residual)

print("\nAfter Layer Normalization:")
print(normalized)
