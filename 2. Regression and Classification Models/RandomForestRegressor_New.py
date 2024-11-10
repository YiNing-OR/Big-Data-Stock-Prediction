import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.errors = []

    def fit(self, X, y):
        np.random.seed(0)  # For reproducibility
        n_samples = len(X)
        cumulative_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples) # Track out-of-bag sample counts for error calculations

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if len(self.errors) == 0:
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                weights = self.errors[-1] / np.sum(self.errors[-1])
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)

            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Create a DecisionTreeRegressor and fit it on the bootstrap sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            # Calculate Out-of-Bag error for current tree
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            if len(oob_indices) > 0:
                oob_predictions = tree.predict(X[oob_indices])
                cumulative_predictions[oob_indices] += oob_predictions
                oob_counts[oob_indices] += 1

            # Update cumulative predictions and compute errors + Average predictions for each data point
            valid_oob_indices = oob_counts > 0
            oob_avg_predictions = np.zeros(len(X))
            oob_avg_predictions[valid_oob_indices] = cumulative_predictions[valid_oob_indices] / oob_counts[
                valid_oob_indices]

            # Calculate the error for the OOB samples only as absolute difference
            errors = np.zeros(len(X))
            errors[valid_oob_indices] = np.abs(y[valid_oob_indices] - oob_avg_predictions[valid_oob_indices])
            self.errors.append(errors)  # Store the errors after each tree

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def rmse(self, X, y):
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))
