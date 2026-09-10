import math


def cross_entropy(probability):

    loss = -math.log(probability)

    return loss


# Example:
# Python probability = 20.22%

probability = 0.2022

loss = cross_entropy(probability)

print("Probability:")
print(probability)

print("\nCross-Entropy Loss:")
print(loss)
