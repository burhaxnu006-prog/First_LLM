import math


def loss(probability):

    return -math.log(probability)


probability = 0.2022

small_change = 0.0001

loss_before = loss(probability)

loss_after = loss(
    probability + small_change
)

gradient = (
    loss_after - loss_before
) / small_change


print("Loss before:")
print(loss_before)

print("\nLoss after:")
print(loss_after)

print("\nGradient:")
print(gradient)
