import numpy as np

class Node:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.feature = None # index kolom yang dipakai untuk split
        self.threshold = None # batas buat split 

        self.missing_go_left = True # buat yang nilai NaN
        self.left = None
        self.right = None

    def is_leaf(self) -> bool:
        return self.left is None # kalo ga ada left node berarti udah leaf

def gini(counts: np.ndarray) -> float | np.ndarray:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1) # buat make sure ga ada pembagian dengan 0

    return 1.0 - np.sum(p ** 2, axis=-1)

def weighted_gini(left_counts: np.ndarray, right_counts: np.ndarray) -> np.ndarray: 
    n_left = left_counts.sum(axis=-1)
    n_right = right_counts.sum(axis=-1)
    return (n_left * gini(left_counts) + n_right * gini(right_counts)) / (n_left + n_right)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth harus >= 1 atau None")
        
        if min_samples_split < 2:
            raise ValueError("min_samples_split harus >= 2")

        self.max_depth = max_depth
        self.min_samples_split= min_samples_split
        self.max_features = max_features # nanti dipakai di random forest
        self.random_state = random_state

        self.classes = None
        self._n_classes = None
        self._n_features = None
        self._is_categorical = None
        self._root = None
        self._n_features_split = None
        self._rng = None

    def _split_numeric(self, x,y, missing_counts: np.ndarray) -> tuple[float, float, bool] | None:
        # buat nyari threshold dari salah satu fitur numerik
        # returnya berupa weighted_gini, threshold, sama missing go left

        if len(x) < 2:
            return None

        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        total = np.bincount(y_sorted, minlength=self._n_classes)
        left = np.zeros(self._n_classes)

        best_gini, best_threshold, nan_left = np.inf, None, None

        for i in range(len(x_sorted) -1):
            left[y_sorted[i]] += 1

            if x_sorted[i] == x_sorted[i + 1]:
                continue

            right = total - left
            threshold = (x_sorted[i] + x_sorted[i + 1]) / 2

            gain_left = weighted_gini(left + missing_counts, right)
            gain_right = weighted_gini(left, right + missing_counts)

            if gain_left < best_gini:
                best_gini, best_threshold, nan_left = gain_left, threshold, True

            if gain_right < best_gini:
                best_gini, best_threshold, nan_left = gain_right, threshold, False

        if best_threshold is None:
            return None

        return float(best_gini), float(best_threshold), nan_left

    def _split_categorical(self, x,y, missing_counts: np.ndarray) -> tuple[float, float, bool] | None:
        # buat nyari kategori terbaik
        # return weighted_gini, kategori, sama missing go left

        categories = np.unique(x)
        if len(categories) < 2:
            return None

        total = np.bincount(y, minlength=self._n_classes)
        best_gini, best_category, nan_left = np.inf, None, None

        for c in categories:
            left = np.bincount(y[x==c], minlength=self._n_classes)
            right = total - left

            gini_left = weighted_gini(left + missing_counts, right)
            gini_right = weighted_gini(left, right + missing_counts)

            if gini_left < best_gini:
                best_gini, best_category, nan_left = gini_left, c, True

            if gini_right< best_gini:
                best_gini, best_category, nan_left = gini_right, c, False

        return float(best_gini), float(best_category), nan_left

    def _best_split(self, X, y: np.ndarray)  -> tuple[int, float, bool, float] | None:
        parent_gini = gini(np.bincount(y, minlength=self._n_classes))

        features = np.arange(self._n_features) # array angka feature dari 0 sampai n - 1
        if self._n_features_split < self._n_features:
            features = self._rng.choice(self._n_features, size = self._n_features_split, replace = False)

        best = None

        for j in features:
            col = X[:, j] # kolom dari fitur j
            missing = np.isnan(col) # data missing 
            missing_counts = np.bincount(y[missing], minlength=self._n_classes) # jumlah missing
            x_cleaned = col[~missing] 
            y_cleaned = y[~missing]

            if self._is_categorical[j]:
                result = self._split_categorical(x_cleaned, y_cleaned, missing_counts)
            else:
                result = self._split_numeric(x_cleaned, y_cleaned, missing_counts)

            if result is None:
                continue

            child_gini, threshold, go_left = result
            if best is None or child_gini < best['child_gini']:
                best = {
                    "feature": int(j),
                    "threshold": threshold,
                    "go_left": go_left,
                    "child_gini": child_gini
                }

        if best is None:
            return None

        return best["feature"], best["threshold"], best["go_left"], float(parent_gini - best["child_gini"])

    def _go_left(self, col: np.ndarray, feature: int, threshold: float, go_left: bool) -> np.ndarray:

        if self._is_categorical[feature]:
            left = col == threshold
        else:
            left = col <= threshold

        left[np.isnan(col)] = go_left

        return left

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # membangun tree secara rekursif

        counts = np.bincount(y, minlength=self._n_classes)
        node = Node(counts / len(y))

        # base case recursive
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node
        if np.count_nonzero(counts) == 1: # node sudah murni
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold, go_left, gain = split
        if gain <= 1e-12:
            return node

        col = X[:, feature]
        left_mask = self._go_left(col, feature, threshold, go_left)

        # kalau node nya ga ada nan, ketika predik nan masuk ke child yang lebih besar
        if not np.isnan(col).any():
            go_left = bool(left_mask.sum() >= len(y) / 2)

        node.feature = feature
        node.threshold = threshold
        node.missing_go_left = go_left
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray, categorical_features: list[int] | None = None) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X harus 2 dimensi (n sampel, n feature)")
        if y.ndim != 1:
            raise ValueError("y harus 1D")
        if len(X) != len(y):
            raise ValueError("jumlah baris X dan y harus sama")
        if len(y) == 0:
            raise ValueError("data kosong")

        self.classes, y_idx = np.unique(y, return_inverse=True)
        self._n_classes = len(self.classes)
        self._n_features = X.shape[1]

        self._is_categorical = np.zeros(self._n_features, dtype=bool)
        if categorical_features is not None:
            self._is_categorical[categorical_features] = True

        if self.max_features is None:
            self._n_features_split = self._n_features
        else:
            self._n_features_split = min(int(self.max_features), self._n_features)

        if self._n_features_split < 1:
            raise ValueError("max features harus >= 1")

        self._rng = np.random.default_rng(self.random_state)
        self._root = self._build(X, y_idx, depth=0)
        return self

    def _traverse(self, node: Node, X: np.ndarray, idx: np.ndarray, prob: np.ndarray) -> None:
        # traverse suatu baris sampai ke leaf

        if len(idx) == 0:
            return
        if node.is_leaf():
            prob[idx] = node.value
            return

        left = self._go_left(X[idx, node.feature], node.feature, node.threshold, node.missing_go_left)
        self._traverse(node.left, X, idx[left], prob)
        self._traverse(node.right, X, idx[~left], prob)

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("model belum pernah di fit")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._n_features:
            raise ValueError(f"X harus 2 dimensi dengan {self._n_features} kolom")

        prob = np.zeros((len(X), self._n_classes))
        self._traverse(self._root, X, np.arange(len(X)), prob)
        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_probability(X), axis=1)]