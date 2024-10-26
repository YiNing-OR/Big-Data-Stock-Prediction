import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Check the stopping criteria
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is not None:
                left_indices = X[:, best_feature] < best_threshold
                right_indices = X[:, best_feature] >= best_threshold
                left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
                right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
                return (best_feature, best_threshold, left_tree, right_tree)

        # Return the mean value if stopping criteria is met
        return np.mean(y)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if np.any(left_indices) and np.any(right_indices):
                    left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                    right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)
                    mse = (len(y[left_indices]) * left_mse + len(y[right_indices]) * right_mse) / len(y)

                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if isinstance(tree, tuple):
            feature, threshold, left_tree, right_tree = tree
            if sample[feature] < threshold:
                return self._predict_sample(sample, left_tree)
            else:
                return self._predict_sample(sample, right_tree)
        else:
            return tree  # Leaf node, return mean value

