import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y):
        self.max_features = X.shape[1] if self.max_features is None else min(self.max_features, X.shape[1])
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_feature, best_threshold, best_mse = self._best_split(X, y)
            
            if best_feature is not None and (np.var(y) - best_mse) > self.min_impurity_decrease: 
                # Special Feature : Early Stopping Based on Impurity Decrease (min_impurity_decrease):
                    # Only splits if the decrease in variance (impurity) is above a certain threshold. 
                    # Checked in _build_tree before continuing with a split.

                left_indices = X[:, best_feature] < best_threshold
                right_indices = X[:, best_feature] >= best_threshold
                
                if np.sum(left_indices) >= self.min_samples_leaf and np.sum(right_indices) >= self.min_samples_leaf:
                    left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
                    right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
                    return (best_feature, best_threshold, left_tree, right_tree)

        # Else, Return the mean value if stopping criteria is met
        return np.mean(y)

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float('inf')
        n_features = X.shape[1]
        
        # Special Feature : Randomly select a subset of features to consider
        features = np.random.choice(n_features, self.max_features, replace=False)
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if np.any(left_indices) and np.any(right_indices):
                    left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
                    right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)
                    weighted_mse = (len(y[left_indices]) * left_mse + len(y[right_indices]) * right_mse) / len(y)

                    if weighted_mse < best_mse:
                        best_mse = weighted_mse
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold, best_mse

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
        
    def score(self, X, y):
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def rmse(self, X, y):
        predictions = self.predict(X)
        return np.sqrt(np.mean((y - predictions) ** 2))
