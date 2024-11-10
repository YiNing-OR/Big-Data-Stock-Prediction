import numpy as np
class RidgeLinearRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, epochs=1000):
        self.alpha = alpha        # Regularization strength
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None       # To store learned coefficients
        self.bias = None          # To store intercept

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for epoch in range(self.epochs):
            # Calculate predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate gradient with L2 penalty
            dw = (2 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias