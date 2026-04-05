"""
Custom Machine Learning Algorithms - Pure Python (stdlib only).
Implements Naive Bayes, Linear Regression, KNN, Decision Tree, and a simple Neural Network.
No numpy or scikit-learn required.
"""

import math
import random
import collections
from copy import deepcopy


# ── Vector / Matrix Utilities ──────────────────────────────────────

def dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def vec_scale(a, s):
    return [x * s for x in a]


def vec_mean(vectors):
    n = len(vectors)
    if n == 0:
        return []
    d = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(d)]


def vec_std(vectors, mean_vec=None):
    n = len(vectors)
    if n <= 1:
        return [1.0] * len(vectors[0]) if vectors else []
    if mean_vec is None:
        mean_vec = vec_mean(vectors)
    d = len(mean_vec)
    result = []
    for i in range(d):
        var = sum((v[i] - mean_vec[i]) ** 2 for v in vectors) / (n - 1)
        result.append(max(math.sqrt(var), 1e-10))
    return result


def normalize(vectors):
    """Z-score normalization. Returns normalized vectors, mean, std."""
    mean_v = vec_mean(vectors)
    std_v = vec_std(vectors, mean_v)
    normed = [[(v[i] - mean_v[i]) / std_v[i] for i in range(len(v))] for v in vectors]
    return normed, mean_v, std_v


def euclidean_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0, x)


def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ── TF-IDF Text Vectorizer ────────────────────────────────────────

class TFIDFVectorizer:
    """Simple TF-IDF vectorizer for text classification."""

    def __init__(self, max_features=500):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}
        self.doc_count = 0

    def fit(self, documents):
        """Build vocabulary and compute IDF from documents."""
        self.doc_count = len(documents)
        term_doc_freq = collections.Counter()
        term_freq_total = collections.Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                term_doc_freq[t] += 1
            for t in tokens:
                term_freq_total[t] += 1

        # Select top features by frequency
        top_terms = [t for t, _ in term_freq_total.most_common(self.max_features)]
        self.vocabulary = {t: i for i, t in enumerate(top_terms)}

        # Compute IDF
        for term, idx in self.vocabulary.items():
            df = term_doc_freq.get(term, 0)
            self.idf[term] = math.log((self.doc_count + 1) / (df + 1)) + 1

        return self

    def transform(self, documents):
        """Transform documents to TF-IDF vectors."""
        vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = collections.Counter(tokens)
            total = len(tokens) if tokens else 1
            vec = [0.0] * len(self.vocabulary)
            for term, idx in self.vocabulary.items():
                if term in tf:
                    tfidf = (tf[term] / total) * self.idf.get(term, 1.0)
                    vec[idx] = tfidf
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def _tokenize(self, text):
        """Simple tokenizer: lowercase, split on non-alpha."""
        import re
        return re.findall(r'[a-z]+', text.lower())


# ── Naive Bayes Classifier ────────────────────────────────────────

class NaiveBayesClassifier:
    """Gaussian Naive Bayes for continuous features."""

    def __init__(self):
        self.classes = []
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}

    def fit(self, X, y):
        """Train on feature vectors X and labels y."""
        class_data = collections.defaultdict(list)
        for xi, yi in zip(X, y):
            class_data[yi].append(xi)

        self.classes = sorted(class_data.keys())
        n_total = len(y)

        for cls in self.classes:
            data = class_data[cls]
            self.class_priors[cls] = len(data) / n_total
            mean = vec_mean(data)
            self.class_means[cls] = mean
            d = len(mean)
            var = []
            for i in range(d):
                v = sum((x[i] - mean[i]) ** 2 for x in data) / max(len(data), 1)
                var.append(max(v, 1e-9))
            self.class_vars[cls] = var

        return self

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def predict_proba(self, X):
        return [self._predict_proba_one(x) for x in X]

    def _predict_proba_one(self, x):
        log_probs = {}
        for cls in self.classes:
            log_p = math.log(self.class_priors[cls])
            for i in range(len(x)):
                mean = self.class_means[cls][i]
                var = self.class_vars[cls][i]
                log_p += -0.5 * math.log(2 * math.pi * var)
                log_p += -0.5 * ((x[i] - mean) ** 2) / var
            log_probs[cls] = log_p

        # Convert to probabilities via softmax
        max_lp = max(log_probs.values())
        exp_probs = {c: math.exp(lp - max_lp) for c, lp in log_probs.items()}
        total = sum(exp_probs.values())
        return {c: p / total for c, p in exp_probs.items()}

    def _predict_one(self, x):
        probs = self._predict_proba_one(x)
        return max(probs, key=probs.get)

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ── Linear Regression ─────────────────────────────────────────────

class LinearRegression:
    """Ordinary Least Squares via gradient descent."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0
        self.mean_x = []
        self.std_x = []
        self.mean_y = 0.0
        self.std_y = 1.0

    def fit(self, X, y):
        n = len(X)
        if n == 0:
            return self
        d = len(X[0])

        # Normalize
        self.mean_x = vec_mean(X)
        self.std_x = vec_std(X, self.mean_x)
        self.mean_y = sum(y) / n
        self.std_y = max(math.sqrt(sum((yi - self.mean_y) ** 2 for yi in y) / max(n - 1, 1)), 1e-10)

        X_norm = [[(X[i][j] - self.mean_x[j]) / self.std_x[j] for j in range(d)] for i in range(n)]
        y_norm = [(yi - self.mean_y) / self.std_y for yi in y]

        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.epochs):
            # Forward
            preds = [dot(X_norm[i], self.weights) + self.bias for i in range(n)]
            errors = [preds[i] - y_norm[i] for i in range(n)]

            # Gradients
            grad_w = [sum(errors[i] * X_norm[i][j] for i in range(n)) / n for j in range(d)]
            grad_b = sum(errors) / n

            # Update
            self.weights = [self.weights[j] - self.lr * grad_w[j] for j in range(d)]
            self.bias -= self.lr * grad_b

        return self

    def predict(self, X):
        d = len(self.weights)
        results = []
        for x in X:
            x_norm = [(x[j] - self.mean_x[j]) / self.std_x[j] for j in range(d)]
            y_norm = dot(x_norm, self.weights) + self.bias
            results.append(y_norm * self.std_y + self.mean_y)
        return results

    def score(self, X, y):
        """R-squared score."""
        preds = self.predict(X)
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
        mean_y = sum(y) / len(y)
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        return 1 - ss_res / max(ss_tot, 1e-10)


# ── Gradient Boosting Regressor (simplified) ──────────────────────

class GradientBoostingRegressor:
    """Simplified gradient boosting with decision stumps."""

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = 0.0

    def fit(self, X, y):
        n = len(y)
        self.initial_pred = sum(y) / n
        current_preds = [self.initial_pred] * n

        for _ in range(self.n_estimators):
            residuals = [y[i] - current_preds[i] for i in range(n)]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            tree_preds = tree.predict(X)
            current_preds = [current_preds[i] + self.lr * tree_preds[i] for i in range(n)]

        return self

    def predict(self, X):
        preds = [self.initial_pred] * len(X)
        for tree in self.trees:
            tree_preds = tree.predict(X)
            preds = [preds[i] + self.lr * tree_preds[i] for i in range(len(X))]
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, preds))
        mean_y = sum(y) / len(y)
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        return 1 - ss_res / max(ss_tot, 1e-10)


# ── K-Nearest Neighbors ───────────────────────────────────────────

class KNNClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        distances = [(euclidean_dist(x, xt), yt) for xt, yt in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda d: d[0])
        k_nearest = [d[1] for d in distances[:self.k]]
        counter = collections.Counter(k_nearest)
        return counter.most_common(1)[0][0]

    def predict_proba(self, X):
        results = []
        for x in X:
            distances = [(euclidean_dist(x, xt), yt) for xt, yt in zip(self.X_train, self.y_train)]
            distances.sort(key=lambda d: d[0])
            k_nearest = [d[1] for d in distances[:self.k]]
            counter = collections.Counter(k_nearest)
            total = sum(counter.values())
            probs = {cls: count / total for cls, count in counter.items()}
            results.append(probs)
        return results

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ── Decision Tree ──────────────────────────────────────────────────

class DecisionTreeClassifier:
    """Simple decision tree classifier using Gini impurity."""

    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        data = list(zip(X, y))
        self.tree = self._build_tree(data, depth=0)
        return self

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def predict_proba(self, X):
        results = []
        for x in X:
            leaf = self._get_leaf(x, self.tree)
            results.append(leaf)
        return results

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)

    def _gini(self, groups, classes):
        n_total = sum(len(g) for g in groups)
        if n_total == 0:
            return 0
        gini = 0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0
            labels = [row[1] for row in group]
            for cls in classes:
                p = labels.count(cls) / size
                score += p * p
            gini += (1.0 - score) * (size / n_total)
        return gini

    def _split(self, data, feature_idx, threshold):
        left = [row for row in data if row[0][feature_idx] <= threshold]
        right = [row for row in data if row[0][feature_idx] > threshold]
        return left, right

    def _best_split(self, data):
        classes = list(set(row[1] for row in data))
        best_idx, best_thresh, best_gini, best_groups = None, None, float('inf'), None
        n_features = len(data[0][0])

        for feat_idx in range(n_features):
            values = sorted(set(row[0][feat_idx] for row in data))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left, right = self._split(data, feat_idx, threshold)
                gini = self._gini([left, right], classes)
                if gini < best_gini:
                    best_idx, best_thresh, best_gini, best_groups = feat_idx, threshold, gini, (left, right)

        return {"feature": best_idx, "threshold": best_thresh, "gini": best_gini, "groups": best_groups}

    def _build_tree(self, data, depth):
        classes = list(set(row[1] for row in data))
        if len(classes) == 1:
            return {"leaf": True, "class": classes[0], "proba": {classes[0]: 1.0}}
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return self._make_leaf(data)

        split = self._best_split(data)
        if split["groups"] is None:
            return self._make_leaf(data)

        left_data, right_data = split["groups"]
        if len(left_data) == 0 or len(right_data) == 0:
            return self._make_leaf(data)

        return {
            "leaf": False,
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": self._build_tree(left_data, depth + 1),
            "right": self._build_tree(right_data, depth + 1),
        }

    def _make_leaf(self, data):
        labels = [row[1] for row in data]
        counter = collections.Counter(labels)
        total = len(labels)
        proba = {cls: count / total for cls, count in counter.items()}
        return {"leaf": True, "class": counter.most_common(1)[0][0], "proba": proba}

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def _get_leaf(self, x, node):
        if node["leaf"]:
            return node["proba"]
        if x[node["feature"]] <= node["threshold"]:
            return self._get_leaf(x, node["left"])
        return self._get_leaf(x, node["right"])


class DecisionTreeRegressor:
    """Decision tree regressor using variance reduction."""

    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        data = list(zip(X, y))
        self.tree = self._build_tree(data, depth=0)
        return self

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _variance(self, data):
        if len(data) <= 1:
            return 0
        vals = [d[1] for d in data]
        mean = sum(vals) / len(vals)
        return sum((v - mean) ** 2 for v in vals) / len(vals)

    def _build_tree(self, data, depth):
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return {"leaf": True, "value": sum(d[1] for d in data) / max(len(data), 1)}

        best_feat, best_thresh, best_score = None, None, float('inf')
        best_left, best_right = None, None
        n_features = len(data[0][0])

        # Sample features for speed
        feature_indices = list(range(n_features))
        if n_features > 10:
            feature_indices = random.sample(feature_indices, min(10, n_features))

        for feat_idx in feature_indices:
            values = sorted(set(d[0][feat_idx] for d in data))
            # Sample thresholds for speed
            if len(values) > 20:
                values = sorted(random.sample(values, 20))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left = [d for d in data if d[0][feat_idx] <= threshold]
                right = [d for d in data if d[0][feat_idx] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = (len(left) * self._variance(left) + len(right) * self._variance(right)) / len(data)
                if score < best_score:
                    best_feat, best_thresh, best_score = feat_idx, threshold, score
                    best_left, best_right = left, right

        if best_feat is None:
            return {"leaf": True, "value": sum(d[1] for d in data) / max(len(data), 1)}

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(best_left, depth + 1),
            "right": self._build_tree(best_right, depth + 1),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])


# ── Random Forest ──────────────────────────────────────────────────

class RandomForestClassifier:
    """Random forest using bootstrap aggregation of decision trees."""

    def __init__(self, n_estimators=10, max_depth=8, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        n = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        results = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(len(self.trees))]
            counter = collections.Counter(votes)
            results.append(counter.most_common(1)[0][0])
        return results

    def predict_proba(self, X):
        all_probas = [tree.predict_proba(X) for tree in self.trees]
        results = []
        for i in range(len(X)):
            combined = collections.defaultdict(float)
            for t in range(len(self.trees)):
                for cls, prob in all_probas[t][i].items():
                    combined[cls] += prob
            total = sum(combined.values())
            results.append({cls: p / total for cls, p in combined.items()})
        return results

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ── Simple Neural Network (MLP) ───────────────────────────────────

class SimpleNeuralNetwork:
    """Multi-layer perceptron with configurable hidden layers."""

    def __init__(self, layer_sizes, learning_rate=0.01, epochs=500):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # Xavier initialization
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            w = [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0] * fan_out
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, x):
        activations = [x]
        current = x
        for i in range(len(self.weights)):
            n_out = len(self.biases[i])
            z = [sum(current[j] * self.weights[i][j][k] for j in range(len(current))) + self.biases[i][k]
                 for k in range(n_out)]
            if i < len(self.weights) - 1:
                current = [relu(zi) for zi in z]
            else:
                current = z  # output layer (linear for regression, softmax for classification)
            activations.append(current)
        return activations

    def fit_regression(self, X, y):
        """Train for regression tasks."""
        for epoch in range(self.epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                activations = self._forward(xi)
                output = activations[-1]
                target = yi if isinstance(yi, list) else [yi]

                # MSE loss gradient
                errors = [output[k] - target[k] for k in range(len(target))]
                total_loss += sum(e ** 2 for e in errors)

                # Backprop
                self._backprop(activations, errors)

        return self

    def fit_classifier(self, X, y, classes):
        """Train for classification tasks."""
        self.classes_ = classes
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                activations = self._forward(xi)
                output = softmax(activations[-1])

                # Cross-entropy gradient
                target_idx = class_to_idx[yi]
                errors = list(output)
                errors[target_idx] -= 1.0

                self._backprop(activations, errors)

        return self

    def _backprop(self, activations, output_errors):
        """Simple backpropagation."""
        n_layers = len(self.weights)
        deltas = [None] * n_layers
        deltas[-1] = output_errors

        # Backward pass
        for i in range(n_layers - 2, -1, -1):
            n_out = len(self.biases[i])
            delta = [0.0] * n_out
            for j in range(n_out):
                err = sum(deltas[i + 1][k] * self.weights[i + 1][j][k]
                          for k in range(len(deltas[i + 1])))
                # ReLU derivative
                if activations[i + 1][j] > 0:
                    delta[j] = err
                else:
                    delta[j] = 0
            deltas[i] = delta

        # Update weights
        for i in range(n_layers):
            for j in range(len(activations[i])):
                for k in range(len(deltas[i])):
                    self.weights[i][j][k] -= self.lr * deltas[i][k] * activations[i][j]
            for k in range(len(deltas[i])):
                self.biases[i][k] -= self.lr * deltas[i][k]

    def predict(self, X):
        results = []
        for x in X:
            output = self._forward(x)[-1]
            results.append(output)
        return results

    def predict_classes(self, X):
        results = []
        for x in X:
            output = softmax(self._forward(x)[-1])
            idx = output.index(max(output))
            results.append(self.classes_[idx])
        return results

    def score(self, X, y):
        preds = self.predict_classes(X)
        return sum(1 for p, t in zip(preds, y) if p == t) / len(y)


# ── Model Evaluation Utilities ─────────────────────────────────────

def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split data into train and test sets."""
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return ([X[i] for i in train_idx], [X[i] for i in test_idx],
            [y[i] for i in train_idx], [y[i] for i in test_idx])


def cross_val_score(model_class, X, y, k=5, **model_kwargs):
    """K-fold cross validation."""
    n = len(X)
    fold_size = n // k
    indices = list(range(n))
    random.shuffle(indices)
    scores = []

    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = [j for j in indices if j not in set(test_idx)]
        X_train = [X[j] for j in train_idx]
        y_train = [y[j] for j in train_idx]
        X_test = [X[j] for j in test_idx]
        y_test = [y[j] for j in test_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))

    return scores


def confusion_matrix(y_true, y_pred, classes):
    """Compute confusion matrix."""
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in cls_to_idx and yp in cls_to_idx:
            matrix[cls_to_idx[yt]][cls_to_idx[yp]] += 1
    return matrix


def classification_report(y_true, y_pred, classes):
    """Generate precision, recall, f1 for each class."""
    cm = confusion_matrix(y_true, y_pred, classes)
    report = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(classes))) - tp
        fn = sum(cm[i][j] for j in range(len(classes))) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        report[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return report
