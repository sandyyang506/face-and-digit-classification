"""Template for a 3 layer feed forward neural network for digit
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy for linear algebra. You may not use torch,
tensorflow, sklearn, jax, or keras for training, gradients, or
prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkDigits` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

from util_digits import load_digits, flatten_images


class ScratchNeuralNetworkDigits:
    """3 layer fully connected network: 784 to h1 to h2 to 10.

    Use any reasonable hidden activation (ReLU, sigmoid, tanh). For the
    output layer, softmax paired with cross entropy loss is typical.
    Document your choices in the report.

    Implementation notes:
      * Store weights as numpy arrays: W1 (784, h1), W2 (h1, h2),
        W3 (h2, 10), plus biases b1, b2, b3.
      * Initialise with small random values (scaled Gaussian, He,
        Xavier) to break symmetry.
      * `forward` should cache intermediate activations so that
        `backward` can compute gradients without re running forward.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 10,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        if seed is not None:
            np.random.seed(seed)

        # using He init since hidden activation is ReLU
        s1 = (2.0 / input_size) ** 0.5
        s2 = (2.0 / hidden1_size) ** 0.5
        s3 = (2.0 / hidden2_size) ** 0.5

        self.W1 = np.random.randn(input_size, hidden1_size) * s1
        self.b1 = np.zeros(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * s2
        self.b2 = np.zeros(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size) * s3
        self.b3 = np.zeros(output_size)

        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass.

        `X` has shape (N, 784). Return shape is (N, 10). You may return
        probabilities (after softmax) or raw logits; keep `predict` and
        `backward` consistent with your choice.
        """
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = np.maximum(self.z1, 0)

        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = np.maximum(self.z2, 0)

        self.z3 = self.a2.dot(self.W3) + self.b3

        # subtract max for numerical stability
        z = self.z3 - self.z3.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        self.probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict:
        """Back propagate loss gradients through the network.

        `X` has shape (N, 784); `y_onehot` has shape (N, 10). Return a
        dict like
        `{"dW1": ..., "db1": ..., "dW2": ..., "db2": ..., "dW3": ..., "db3": ...}`.
        """
        n = X.shape[0]

        # softmax + cross entropy gradient simplifies to (probs - y)
        dz3 = (self.probs - y_onehot) / n
        dW3 = self.a2.T.dot(dz3)
        db3 = dz3.sum(axis=0)

        da2 = dz3.dot(self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T.dot(dz2)
        db2 = dz2.sum(axis=0)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T.dot(dz1)
        db1 = dz1.sum(axis=0)

        return {
            "dW1": dW1, "dW2": dW2, "dW3": dW3,
            "db1": db1, "db2": db2, "db3": db3,
        }

    def update_weights(self, grads: dict) -> None:
        """Apply one gradient descent step using `grads` from `backward`."""
        lr = self.learning_rate
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]
        self.W3 -= lr * grads["dW3"]
        self.b3 -= lr * grads["db3"]

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with values in {0..9}.
        """
        X = flatten_images(training_images)
        y = training_labels
        n = len(X)

        for epoch in range(self.num_epochs):
            # reshuffle every epoch
            order = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                xb = X[idx]
                yb = y[idx]

                # one hot encode this batch
                yhot = np.zeros((len(idx), self.output_size))
                yhot[np.arange(len(idx)), yb] = 1

                self.forward(xb)
                grads = self.backward(xb, yhot)
                self.update_weights(grads)

    def predict(self, image: np.ndarray) -> int:
        """Predict a single label in {0..9} for a 28x28 image."""
        x = image.flatten().reshape(1, -1)
        probs = self.forward(x)
        return int(probs.argmax())

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        X = flatten_images(images)
        probs = self.forward(X)
        preds = probs.argmax(axis=1)
        return float((preds == labels).mean())


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on digits."""
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkDigits()
        start = time.time()
        net.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = net.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Scratch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)