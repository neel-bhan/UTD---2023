import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # Stopping criteria
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return unique_labels[0]

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        # Split dataset
        left_idx = X[:, best_feature] < best_threshold
        right_idx = ~left_idx

        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                gini = self._gini_impurity(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold

        return best_feature, best_threshold

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            _, counts = np.unique(y, return_counts=True)
            probs = counts / counts.sum()
            return 1 - np.sum(probs ** 2)

        left_weight = len(left_y) / (len(left_y) + len(right_y))
        right_weight = 1 - left_weight

        return left_weight * gini(left_y) + right_weight * gini(right_y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left_subtree, right_subtree = node
        return self._traverse_tree(x, left_subtree if x[feature] < threshold else right_subtree)


# Example usage
if __name__ == "__main__":
    X = np.array([[2.7], [1.5], [3.2], [5.0], [4.5], [6.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)
    predictions = tree.predict(X)
    print("Predictions:", predictions)
