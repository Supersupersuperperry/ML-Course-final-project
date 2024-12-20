import numpy as np

class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, regularization=0.0, verbose=False):
        # learning_rate: step size for gradient descent
        # max_iter: number of iterations for training
        # regularization: L2 regularization strength (lambda)
        # verbose: whether to print training progress
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.verbose = verbose
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    def compute_loss_and_gradient(self, X, y):
        # Compute predictions
        z = X.dot(self.weights) + self.bias
        preds = self.sigmoid(z)

        # To avoid log(0), add epsilon
        epsilon = 1e-9
        loss = -np.mean(y * np.log(preds + epsilon) + (1 - y) * np.log(1 - preds + epsilon))
        
        # Add L2 regularization to the loss (excluding bias)
        if self.regularization > 0:
            loss += (self.regularization / (2 * X.shape[0])) * np.sum(self.weights**2)

        # Compute gradients
        grad_w = (1/X.shape[0]) * X.T.dot(preds - y)
        if self.regularization > 0:
            grad_w += (self.regularization / X.shape[0]) * self.weights
        grad_b = np.mean(preds - y)

        return loss, grad_w, grad_b

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.max_iter):
            loss, grad_w, grad_b = self.compute_loss_and_gradient(X, y)

            # Update weights and bias
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}/{self.max_iter}, Loss: {loss:.4f}")

    def predict(self, X):
        z = X.dot(self.weights) + self.bias
        preds = self.sigmoid(z)
        return (preds >= 0.5).astype(int)
