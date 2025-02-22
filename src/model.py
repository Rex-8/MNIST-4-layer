import numpy as np
import os

class NeuralNetwork:
    def __init__(self, load_from_dir=None):
        self.input_size = 784
        self.hidden1_size = 16
        self.hidden2_size = 16
        self.output_size = 10
        if load_from_dir:
            self.load_parameters(load_from_dir)
        else:
            self.initialize_parameters()

    def initialize_parameters(self):
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(2.0 / self.input_size)
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0 / self.hidden1_size)
        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(2.0 / self.hidden2_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        self.b2 = np.zeros((1, self.hidden2_size))
        self.b3 = np.zeros((1, self.output_size))

    def load_parameters(self, load_dir):
        self.W1 = np.load(os.path.join(load_dir, "W1.npy"))
        self.W2 = np.load(os.path.join(load_dir, "W2.npy"))
        self.W3 = np.load(os.path.join(load_dir, "W3.npy"))
        self.b1 = np.load(os.path.join(load_dir, "b1.npy"))
        self.b2 = np.load(os.path.join(load_dir, "b2.npy"))
        self.b3 = np.load(os.path.join(load_dir, "b3.npy"))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.relu(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.softmax(Z3)
        return A1, A2, A3, Z1, Z2, Z3

    def compute_loss(self, A3, Y):
        num_samples = A3.shape[0]
        loss = -np.sum(Y * np.log(A3 + 1e-15)) / num_samples
        return loss

    def backward_propagation(self, X, Y, A1, A2, A3, Z1, Z2, Z3):
        num_samples = X.shape[0]
        dZ3 = A3 - Y
        dW3 = np.dot(A2.T, dZ3) / num_samples
        db3 = np.sum(dZ3, axis=0, keepdims=True) / num_samples
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / num_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / num_samples
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / num_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / num_samples
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
        return gradients

    def update_parameters(self, gradients, learning_rate=0.01, l2_lambda=0.0):
        self.W1 -= learning_rate * (gradients["dW1"] + l2_lambda * self.W1)
        self.W2 -= learning_rate * (gradients["dW2"] + l2_lambda * self.W2)
        self.W3 -= learning_rate * (gradients["dW3"] + l2_lambda * self.W3)
        self.b1 -= learning_rate * gradients["db1"]
        self.b2 -= learning_rate * gradients["db2"]
        self.b3 -= learning_rate * gradients["db3"]

if __name__ == "__main__":
    nn = NeuralNetwork()
    print("W1 shape:", nn.W1.shape)
    print("W2 shape:", nn.W2.shape)
    print("W3 shape:", nn.W3.shape)
    print("b1 shape:", nn.b1.shape)
    print("b2 shape:", nn.b2.shape)
    print("b3 shape:", nn.b3.shape)
    print("Sample W1 values:", nn.W1[0, :5])
    print("Sample b1 values:", nn.b1[0, :5])