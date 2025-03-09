import numpy as np


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wi = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wo = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01
        self.Wc = np.random.randn(hidden_dim, hidden_dim + input_dim) * 0.01

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x_t, h_prev, c_prev):
        concat = np.vstack((h_prev, x_t))

        f_t = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i_t = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        o_t = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

        c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t


class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim

        self.lstm_cell = LSTMCell(input_dim, hidden_dim)
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.by = np.zeros((output_dim, 1))

    def forward(self, X):
        h_t = np.zeros((self.hidden_dim, 1))
        c_t = np.zeros((self.hidden_dim, 1))

        for x_t in X:
            h_t, c_t = self.lstm_cell.forward(x_t, h_t, c_t)

        y_hat = np.dot(self.Wy, h_t) + self.by
        return y_hat


# Example usage
if __name__ == "__main__":
    # Sample input sequence (3 timesteps, input size 2)
    X = [np.random.randn(2, 1) for _ in range(3)]

    lstm = LSTM(input_dim=2, hidden_dim=4, output_dim=1)
    output = lstm.forward(X)
    print("LSTM Output:", output)
