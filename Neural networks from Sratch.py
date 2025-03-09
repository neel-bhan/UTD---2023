import numpy as np


# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', learning_rate=0.1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i + 1])))

        # Set activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)
        return activations, zs

    def backward(self, X, y, activations, zs):
        m = X.shape[0]
        deltas = [activations[-1] - y]

        # Backpropagation
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        # Gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            activations, zs = self.forward(X)
            self.backward(X, y, activations, zs)
            if epoch % 100 == 0:
                loss = np.mean((activations[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]


# Example usage
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(layers=[2, 4, 1], activation='sigmoid', learning_rate=0.1)
    nn.train(X, y, epochs=10000)

    predictions = nn.predict(X)
    print("Predictions:")
    print(predictions)