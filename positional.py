class PositionalEmbedding:

    def __init__(self, max_length, embedding_size):
        self.max_length = max_length
        self.embedding_size = embedding_size

        # Simple position vectors
        self.positions = []

        for position in range(max_length):

            vector = []

            for dimension in range(embedding_size):
                value = position / max_length
                vector.append(value)

            self.positions.append(vector)

    def get_position(self, position):
        return self.positions[position]


# Example settings
max_length = 10
embedding_size = 4

position_embedding = PositionalEmbedding(
    max_length,
    embedding_size
)


# Get position 3
for position in range(5):

    vector = position_embedding.get_position(position)

    print("Position", position, "→", vector)
