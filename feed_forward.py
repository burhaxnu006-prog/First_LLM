import random


def matrix_multiply(vector, matrix):

    result = []

    for column in range(len(matrix[0])):

        total = 0

        for row in range(len(vector)):
            total += vector[row] * matrix[row][column]

        result.append(total)

    return result


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


class FeedForward:

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer weights
        self.w1 = []

        for _ in range(input_size):

            row = []

            for _ in range(hidden_size):
                row.append(random.uniform(-1, 1))

            self.w1.append(row)

        # First layer bias
        self.b1 = [
            0
            for _ in range(hidden_size)
        ]

        # Second layer weights
        self.w2 = []

        for _ in range(hidden_size):

            row = []

            for _ in range(input_size):
                row.append(random.uniform(-1, 1))

            self.w2.append(row)

        # Second layer bias
        self.b2 = [
            0
            for _ in range(input_size)
        ]

    def forward(self, x):

        # First linear layer
        hidden = matrix_multiply(
            x,
            self.w1
        )

        hidden = add_vectors(
            hidden,
            self.b1
        )

        print("\nAfter first linear layer:")
        print(hidden)

        # ReLU activation
        hidden = relu(hidden)

        print("\nAfter ReLU:")
        print(hidden)

        # Second linear layer
        output = matrix_multiply(
            hidden,
            self.w2
        )

        output = add_vectors(
            output,
            self.b2
        )

        print("\nFFN output:")
        print(output)

        return output


# --------------------------------
# Example
# --------------------------------

input_size = 2
hidden_size = 4

ffn = FeedForward(
    input_size,
    hidden_size
)

x = [0.8, 0.6]

print("Input:")
print(x)

ffn.forward(x)
