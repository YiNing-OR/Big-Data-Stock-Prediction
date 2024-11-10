import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, num_iterations=1000, learning_rate=0.01):
        self.alpha = alpha            # Regularization parameter
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Initialize weights
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Gradient descent optimization
        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.alpha * np.sign(self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias