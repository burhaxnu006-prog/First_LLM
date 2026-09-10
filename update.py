# -------------------------
# WEIGHT UPDATE
# -------------------------

weight = 0.50

gradient = -0.20

learning_rate = 0.01


print("Old weight:")
print(weight)


change = learning_rate * gradient

print("\nWeight change:")
print(change)


weight = weight - change


print("\nNew weight:")
print(weight)
