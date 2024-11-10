import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from DecisionTreeRegressor import DecisionTreeRegressor

class XGBoostRegressor:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3, gamma=0, lambda_=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.trees = []

    def fit(self, X, y):
        # Initialize predictions with zeros
        self.y_pred = np.zeros(y.shape)

        for _ in range(self.n_estimators):
            # Calculate the residuals
            residuals = y - self.y_pred
            
            # Fit a decision tree to the residuals with L2 regularization
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals - self.lambda_ * self.y_pred)

            # Update predictions with gamma threshold
            update = tree.predict(X)
            if np.sum((self.learning_rate * update) ** 2) >= self.gamma:
                self.y_pred += self.learning_rate * update

            # Store the tree
            self.trees.append(tree)

    def predict(self, X):
        # Initialize predictions with zeros
        y_pred = np.zeros(X.shape[0])

        # Sum the predictions from each tree
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def rmse(self, X, y):
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))

def feature_engineering(X):

    poly = PolynomialFeatures(degree=2, include_bias=False) # Adding Polynomial Features: Adding polynomial terms for nonlinear relationships.

    X_poly = poly.fit_transform(X) # Log Transformation: Applying a log transformation on skewed features to normalize distributions.
    
    X_log = np.log1p(X)  # Adding 1 to avoid log(0) issues

    # Combine original features, polynomial, and log features
    X_combined = np.hstack((X, X_poly, X_log))
    return X_combined