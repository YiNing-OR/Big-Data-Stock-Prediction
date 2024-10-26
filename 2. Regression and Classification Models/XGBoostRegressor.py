import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class XGBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize predictions with zeros
        self.y_pred = np.zeros(y.shape)

        for _ in range(self.n_estimators):
            # Calculate the residuals
            residuals = y - self.y_pred

            # Fit a decision tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            update = tree.predict(X)
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
