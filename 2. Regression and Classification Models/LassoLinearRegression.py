import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, tol=0.01, max_iter=1000):
        self.alpha = alpha  # Regularization parameter
        self.tol = tol      # Tolerance for convergence
        self.max_iter = max_iter  # Maximum iterations for convergence
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]  # Adds bias (intercept)
        
        # Initialize weights
        self.weights = np.zeros(m + 1)  

        # Coordinate Descent
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Update intercept (weight at index 0)
            self.weights[0] = np.mean(y - X_b[:, 1:] @ self.weights[1:])

            # Update each coefficient using coordinate descent
            for j in range(1, m + 1):
                # Residual calculation without the contribution of feature j
                residual = y - (X_b @ self.weights - X_b[:, j] * self.weights[j])
                
                # Calculate rho for soft thresholding
                rho = X_b[:, j].T @ residual
                
                # Apply soft thresholding to update weights[j]
                if rho < -self.alpha / 2:
                    self.weights[j] = (rho + self.alpha / 2) / (X_b[:, j].T @ X_b[:, j])
                elif rho > self.alpha / 2:
                    self.weights[j] = (rho - self.alpha / 2) / (X_b[:, j].T @ X_b[:, j])
                else:
                    self.weights[j] = 0
            
            # Check for convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break

        # Set intercept and weights separately for clarity
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


def run_lasso_regression(X, y, alpha=1.0, tol=1e-4, max_iter=1000):
    """Fit the Lasso model."""
    model = LassoRegression(alpha=alpha, tol=tol, max_iter=max_iter)
    model.fit(X.values, y.values.ravel())
    return model
