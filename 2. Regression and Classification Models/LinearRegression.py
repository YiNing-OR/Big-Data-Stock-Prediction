import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.intercept = None

    """Adds a bias term (intercept) to the feature matrix X by creating X_b. 
    Calculates the weights (coefficients) using the Normal Equation: 
    weights = ((X^T * X)^-1) * (X^T * y)"""
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adds bias (intercept)
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]  

    """Takes in X, which is input for prediction, and makes predictions 
    by multiplying feature matrix with model coefficients derived in fit()"""
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        return X_b @ np.r_[self.intercept, self.weights]

    """Calculating R² score, which ranges from 0 to 1."""
    def score(self, X, y):
        predictions = self.predict(X)
        residuals = y - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    """Calculating RMSE score, which is giving the difference between predicted and target values."""
    def rmse(self, X, y):
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))


def run_linear_regression(X, y):
    """Fit the model."""
    model = LinearRegression()
    model.fit(X.values, y.values.ravel())
    return model
