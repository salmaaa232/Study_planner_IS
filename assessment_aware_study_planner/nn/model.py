"""Small NumPy neural network with projected gradient updates.

19 inputs -> 24 ReLU units -> one linear score, clipped to [0, 100].
Nonnegative click-input and output weights guarantee nondecreasing marks
when clicks increase, holding all other inputs fixed. Other weights are free.
"""
from pathlib import Path
import numpy as np


class MarkNetwork:
    def __init__(self, hidden=24, seed=42):
        self.hidden, self.seed = hidden, seed

    @staticmethod
    def transform(x):
        x = np.asarray(x, dtype=float).copy()
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != 19 or not np.all(np.isfinite(x)):
            raise ValueError('Expected 19 finite input features')
        if (x[:, :12] < 0).any():
            raise ValueError('Clicks cannot be negative')
        x[:, :12] = np.log1p(x[:, :12])
        return x

    def fit(self, x, y, epochs=100, batch_size=128, learning_rate=0.003, validation=None):
        rng = np.random.default_rng(self.seed)
        x = self.transform(x)
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1
        x = (x - self.mean) / self.scale
        y = np.asarray(y, dtype=float) / 100
        self.w1 = rng.normal(0, 0.12, (x.shape[1], self.hidden))
        self.w1[:12] = np.abs(self.w1[:12]) * 0.15
        self.b1 = np.zeros(self.hidden)
        self.w2 = np.full(self.hidden, 0.02)
        self.b2 = np.array([y.mean()])
        params = [self.w1, self.b1, self.w2, self.b2]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        best, best_weights, stale, step = np.inf, None, 0, 0
        self.history = []
        for epoch in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                ix = order[start:start + batch_size]
                xb, yb = x[ix], y[ix]
                z = xb @ self.w1 + self.b1
                h = np.maximum(z, 0)
                error = h @ self.w2 + self.b2[0] - yb
                dy = 2 * error / len(ix)
                dh = dy[:, None] * self.w2 * (z > 0)
                gradients = [xb.T @ dh + 1e-4 * self.w1, dh.sum(axis=0),
                             h.T @ dy + 1e-4 * self.w2, np.array([dy.sum()])]
                step += 1
                for i, (parameter, gradient) in enumerate(zip(params, gradients)):
                    first[i] = .9 * first[i] + .1 * gradient
                    second[i] = .999 * second[i] + .001 * gradient ** 2
                    parameter -= learning_rate * (first[i] / (1 - .9 ** step)) / (
                        np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8)
                # Projection is the only change from standard Adam training.
                np.maximum(self.w1[:12], 0, out=self.w1[:12])
                np.maximum(self.w2, 0, out=self.w2)
            if validation is not None:
                score = float(np.mean((self.predict(validation[0]) - validation[1]) ** 2))
            else:
                score = float(np.mean((np.clip(np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2[0], 0, 1) - y) ** 2) * 10000)
            self.history.append(dict(epoch=epoch + 1, validation_mse=score))
            if score < best - 1e-5:
                best, stale = score, 0
                best_weights = [p.copy() for p in params]
            else:
                stale += 1
            if stale >= 15:
                break
        self.w1, self.b1, self.w2, self.b2 = best_weights
        return self

    def predict(self, x):
        x = (self.transform(x) - self.mean) / self.scale
        return np.clip((np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2[0]) * 100, 0, 100)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2, seed=self.seed)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as state:
            for name in ['mean', 'scale', 'w1', 'b1', 'w2', 'b2']:
                setattr(model, name, state[name])
            model.seed = int(state['seed'])
        return model
