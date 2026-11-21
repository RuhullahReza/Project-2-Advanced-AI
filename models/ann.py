import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.00, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 -s)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2

ACTIVATIONS = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative)
}

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-np.mean(np.sum(y * np.log(prob), axis=1)))

class NeuralNetworkClassifier:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, int] = (64,32),
        activation: str = "relu",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        random_state: int | None = None,
        verbose: bool = False
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation harus salah satu dari {list(ACTIVATIONS)}")

        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose

        self.classes = None
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.loss_history: list[np.ndarray] = []

    def _init_params(self, n_features, n_classes: int, rng: np.random.Generator) -> None:
        sizes = [n_features, *self.hidden_layer_sizes, n_classes]
        self.weights = []
        self.biases = []

        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / n_in)
            self.weights.append(rng.normal(0.0, std, size=(n_in,n_out)))
            self.biases.append(np.zeros(n_out))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activation, _ = ACTIVATIONS[self.activation]

        A = X
        zs = []
        activations = [X]

        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = activation(Z)
            zs.append(Z)
            activations.append(A)

        Z = A @ self.weights[-1] + self.biases[-1]

        return softmax(Z), zs, activations

    def _backward(
            self,
            Y: np.ndarray,
            prob: np.ndarray,
            zs: list[np.ndarray],
            activations: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        _, activation_derivative = ACTIVATIONS[self.activation]

        m = len(Y)

        grad_weight = [None] * len(self.weights)
        grad_bias = [None] * len(self.biases)

        dZ = (prob - Y) / m

        for i in range(len(self.weights) - 1, -1, -1):
            grad_weight[i] = activations[i].T @ dZ
            grad_bias[i] = dZ.sum(axis=0)

            if i > 0:
                dA = dZ @ self.weights[i].T
                dZ = dA * activation_derivative(zs[i - 1])

        return grad_weight, grad_bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X harus 2D dan jumlah barisnya sama dengan y")

        if np.isnan(X).any():
            raise ValueError("X masih ada nilai NaN")
        
        self.classes, y_idx = np.unique(y, return_inverse=True)

        Y = np.eye(len(self.classes))[y_idx]

        rng = np.random.default_rng(self.random_state)
        self._init_params(X.shape[1], len(self.classes), rng)
        self.loss_history = []

        n = len(X)

        for epoch in range(self.epochs):
            order = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                batch = order[start:start + self.batch_size]

                prob, zs, activations = self._forward(X[batch])
                grads_weight, grads_bias = self._backward(Y[batch], prob, zs, activations)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * grads_weight[i]
                    self.biases[i] -= self.learning_rate * grads_bias[i]

            prob, _, _ = self._forward(X)
            loss = cross_entropy(Y, prob)

            self.loss_history.append(loss)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1}: loss {loss:.4f}")

        return self

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("Model belum pernah di fit")

        prob, _, _ = self._forward(np.asarray(X, dtype=float))
        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_probability(X), axis=1)]