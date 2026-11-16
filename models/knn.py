import numpy as np

# distance
def manhattan_distance(X_test, X_train) -> np.ndarray:
    return np.array([np.abs(q - X_train).sum(axis=1) for q in X_test])

def euclidean_distance(X_test, X_train) -> np.ndarray:
    return np.array([np.sqrt(((q - X_train) ** 2).sum(axis=1)) for q in X_test])

class KNNClassifier:
    def __init__(self, k: int = 3,  distance_metric: str = "euclidean", normalisasi: str = None):
        if k < 1:
            raise ValueError("k harus >= 1")
        self.k = k
        self.distance_metric = distance_metric
        self.normalisasi = normalisasi
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self._fit_normalisasi(X)            
        self.X_train = self._normalisasi_data(X) 
        self.y_train = np.asarray(y)
        return self
    
    def _fit_normalisasi(self, X):
        if self.normalisasi == "minmax":
            self.center = np.nanmin(X, axis=0)
            self.scale = np.nanmax(X, axis=0) - self.center
        elif self.normalisasi == "zscore":
            self.center = np.nanmean(X, axis=0)
            self.scale = np.nanstd(X, axis=0)
        else:
            self.center = np.zeros(X.shape[1])
            self.scale = np.ones(X.shape[1])
        self.scale[self.scale == 0] = 1  # agar tidak 0

        # NaN diisi rata-rata kolom dari data train
        self.isi_nan = np.nanmean(X, axis=0)

    def _normalisasi_data(self, X):
        X = np.where(np.isnan(X), self.isi_nan, X)  # isi NaN dulu
        return (X - self.center) / self.scale
    
    def _compute_distances(self, X_test) -> np.ndarray:

        if self.distance_metric == "manhattan":
            return manhattan_distance(X_test, self.X_train)
        elif self.distance_metric == "euclidean":
            return euclidean_distance(X_test, self.X_train)
        else:
            raise ValueError(f"matriks tidak ada: {self.distance_metric}")

    def predict(self, X_test) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=float)
        X_test = self._normalisasi_data(X_test)    
        D = self._compute_distances(X_test)
        idx = np.argsort(D, axis=1)[:, :self.k]
        label_tetangga = self.y_train[idx]
        hasil = []
        for label in label_tetangga:
            nilai, jumlah = np.unique(label, return_counts=True)
            hasil.append(nilai[np.argmax(jumlah)])
        return np.array(hasil)

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == np.asarray(y_test))