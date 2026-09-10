import math

# -------------------------
# TRAINING STEP
# -------------------------

weight = 0.50

target = 1

learning_rate = 0.01

# Model prediction
logit = weight

# Sigmoid converts logit to probability
probability = 1 / (1 + math.exp(-logit))

print("Weight:")
print(weight)

print("\nProbability:")
print(probability)

# Binary cross-entropy loss
loss = -(
    target * math.log(probability)
    + (1 - target) * math.log(1 - probability)
)

print("\nLoss:")
print(loss)

# Gradient
gradient = probability - target

print("\nGradient:")
print(gradient)

# Weight update
weight = weight - learning_rate * gradient

print("\nUpdated Weight:")
print(weight)
