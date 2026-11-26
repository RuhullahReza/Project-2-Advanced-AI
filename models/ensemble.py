import numpy as np

# cara voting:
# voting="hard", weights=None  -> majority voting (1 model = 1 suara)
# voting="hard", weights=[] -> weighted voting (suara model dikali bobotnya)
# voting="soft", weights=None  -> soft voting (rata-rata probabilitas)
# voting="soft", weights=[] -> weighted soft voting (rata-rata probabilitas berbobot)
class Ensemble:
    def __init__(self,
                 models: list[tuple[str, object]],
                 voting: str = "hard",
                 weights: list[float] | None = None,
                 prefit: bool = False,
                 ) -> None:
        if voting not in ("hard", "soft"):
            raise ValueError("voting harus 'hard' atau 'soft'")

        if weights is not None and len(weights) != len(models):
            raise ValueError("jumlah weights harus sama dengan jumlah model")

        self.models = models # list berisi (nama, model)
        self.voting = voting
        self.weights = weights
        self.prefit = prefit # True kalau model sudah di train sebelumnya

        self.classes: np.ndarray | None = None

    def _get_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.ones(len(self.models)) # semua model bobotnya sama

        return np.asarray(self.weights, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ensemble":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y) # label

        if not self.prefit:
            for name, model in self.models:
                model.fit(X, y) # train tiap model

        return self

    def _model_probability(self, model: object, X: np.ndarray) -> np.ndarray:
        # logistic regression pakai nama predict_probabilities
        if hasattr(model, "predict_probability"):
            prob = model.predict_probability(X)
        else:
            prob = model.predict_probabilities(X)

        # urutan kolom kelas disamakan dengan self.classes
        hasil = np.zeros((len(X), len(self.classes)))

        for i, label in enumerate(model.classes):
            k = np.where(self.classes == label)[0][0] # posisi label di self.classes
            hasil[:, k] = prob[:, i]

        return hasil

    def _soft_vote(self, X: np.ndarray) -> np.ndarray:
        # rata-rata probabilitas semua model, dikali bobot
        weights = self._get_weights()
        total = np.zeros((len(X), len(self.classes)))

        for (name, model), w in zip(self.models, weights):
            total += w * self._model_probability(model, X)

        return total / weights.sum()

    def _hard_vote(self, X: np.ndarray) -> np.ndarray:
        # jumlah suara tiap kelas, suara tiap model dikali bobot
        weights = self._get_weights()
        votes = np.zeros((len(X), len(self.classes)))

        for (name, model), w in zip(self.models, weights):
            pred = model.predict(X)

            for k, label in enumerate(self.classes):
                votes[:, k] += w * (pred == label) # tambah suara kalau model memilih label ini

        return votes

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("model belum pernah di fit")

        X = np.asarray(X, dtype=float)

        if self.voting == "soft":
            return self._soft_vote(X)

        return self._hard_vote(X) / self._get_weights().sum() # proporsi suara tiap kelas

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("model belum pernah di fit")

        X = np.asarray(X, dtype=float)

        if self.voting == "soft":
            prob = self._soft_vote(X)
            return self.classes[np.argmax(prob, axis=1)]

        votes = self._hard_vote(X)
        prob = self._soft_vote(X) # dipakai kalau suaranya seri

        hasil = []
        for i in range(len(X)):
            seri = np.isclose(votes[i], votes[i].max()) # kelas dengan suara terbanyak, bisa lebih dari 1

            # kalau seri (misal 2-2-1), pilih kelas seri yang probabilitasnya paling tinggi
            skor = np.where(seri, prob[i], -1)
            hasil.append(np.argmax(skor))

        return self.classes[np.array(hasil)]
