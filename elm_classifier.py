import numpy as np

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation=sigmoid_activation, random_state=42):
        np.random.seed(random_state)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.random_state = random_state

        # initialize weights
        self.input_weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)

    def fit(self, X, y):
        H = self.activation(X @ self.input_weights + self.bias)
        self.output_weights = np.linalg.pinv(H) @ y

    def predict(self, X):
        H = self.activation(X @ self.input_weights + self.bias)
        output = H @ self.output_weights
        return np.argmax(output, axis=1)
