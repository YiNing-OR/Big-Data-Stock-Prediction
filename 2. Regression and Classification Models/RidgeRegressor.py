import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Reshape y to be a column vector (N, 1)

        # Add regularization term to the identity matrix
        m = X.shape[1]
        I = np.eye(m)

        # Calculate weights using the closed-form solution with regularization
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        # Ensure X is a numpy array
        X = np.array(X)
        
        # Predict using the learned weights
        return X @ self.weights
