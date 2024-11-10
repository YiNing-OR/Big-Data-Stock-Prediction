import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Regularization parameter
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adds bias (intercept)
        identity_matrix = np.eye(X_b.shape[1])  # Identity matrix for regularization
        identity_matrix[0, 0] = 0  # Do not regularize the intercept term
        self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * identity_matrix) @ X_b.T @ y
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ np.r_[self.intercept, self.weights]

    def score(self, X, y):
        predictions = self.predict(X)
        residuals = y - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def rmse(self, X, y):
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))


def run_ridge_regression(X, y, alpha=1.0):
    """Fit the Ridge model."""
    model = RidgeRegression(alpha=alpha)
    model.fit(X.values, y.values.ravel())
    return model
