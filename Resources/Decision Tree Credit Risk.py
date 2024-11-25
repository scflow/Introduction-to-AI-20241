import pandas as pd
import numpy as np

# Load the dataset
# TODO
data = pd.read_csv(?)

# Inspect the first few rows of the dataset to understand its structure and get an overview of the data
print(data.head())

# Assuming the target variable is named 'Risk' and the features are the rest of the columns
# X contains all the feature columns, while y contains the target variable ('Risk')
X = data.drop('Risk', axis=1).values
y = data['Risk'].values

# Split the dataset into training and testing sets
# 70% of the data will be used for training and 30% for testing
# random_state ensures reproducibility of the results
def train_test_split(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a Node class to represent each node in the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Define the Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        # If all labels are the same or max depth is reached, create a leaf node
        if len(unique_labels) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_features)

        # If no valid split is found, create a leaf node
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        split_feature, split_threshold = None, None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_index
                    split_threshold = threshold

        return split_feature, split_threshold

    def _information_gain(self, X, y, feature_index, threshold):
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        n = len(y)
        n_left, n_right = len(y[left_indices]), len(y[right_indices])
        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        #TODO
        return ?

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.value is not None:
            return node.value

        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

# Initialize the Decision Tree Classifier
# This classifier will be used to fit the model to the training data
#TODO
clf = DecisionTreeClassifier(max_depth=?)

# Train the classifier using the training data
clf.fit(X_train, y_train)

# Make predictions on the test set using the trained classifier
y_pred = clf.predict(X_test)

# Evaluate the model's performance using accuracy and classification report
# Accuracy is the ratio of correctly predicted instances to the total instances
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)

# Generate a classification report
from sklearn.metrics import classification_report
print("Classification Report:\n", classification_report(y_test, y_pred))
