import numpy as np

from models.decission_tree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 max_features: int | float | str | None = "sqrt",
                 bootstrap: bool =True,
                 random_state: int | None = None
                 ) -> None:
        
        if n_estimators < 1:
            raise ValueError("n_estimator harus >= 1")

        self.n_estimators = n_estimators # jumlah tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.classes = None
        self.trees: list[DecisionTreeClassifier] = []
        self._n_features = None

    def _resolve_max_features(self, n_features: int) -> int:
        # buat ubah max feature jadi integer
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return min(int(self.max_features), n_features)

    def fit(self, X, y: np.ndarray, categorical_features: list[int] | None = None) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X harus 2 dimensi dan jumlah baris sama dengan y")

        n_samples, self._n_features = X.shape
        self.classes = np.unique(y)

        k = self._resolve_max_features(self._n_features)

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)

            seed = int(rng.integers(0, 2**31 - 1))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=k,
                random_state=seed
            )

            tree.fit(X[idx], y[idx], categorical_features=categorical_features)
            self.trees.append(tree)

        return self

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("model belum pernah di fit")

        X = np.asarray(X, dtype=float)
        prob = np.zeros((len(X), len(self.classes)))

        for tree in self.trees:
            cols = np.searchsorted(self.classes, tree.classes)
            prob [:, cols] += tree.predict_probability(X)

        return prob / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_probability(X), axis=1)]

     
