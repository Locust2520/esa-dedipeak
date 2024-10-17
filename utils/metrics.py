# MIT License

# Copyright (c) 2021 THUML @ Tsinghua University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def peaks_mask(x: np.ndarray, n: int = 6):
    """
    Returns a boolean array of the same shape as x
    """
    middle = x[..., n:-n]
    mean = np.mean(x, axis=-1, keepdims=True)
    s = x.shape[-1]
    peaks = np.all([middle >= x[..., n - i:s - n - i] for i in range(1, n + 1)], axis=0) & \
            np.all([middle >= x[..., n + i:s - n + i] for i in range(1, n + 1)], axis=0) & \
            (middle > mean)

    # extend to the same shape as x
    zeros = np.zeros_like(x[..., 0:n], dtype=bool)
    peaks = np.concatenate([zeros, peaks, zeros], axis=-1)
    return peaks

def pnorm(y_pred: np.ndarray, y_true: np.ndarray, p: int = 2):
    """
    p = 1: MAE
    p = 2: MSE
    """
    peaks = peaks_mask(y_true)
    if p == 1:
        norm = np.mean(np.abs(y_true[peaks] - y_pred[peaks]), axis=-1)
    elif p == 2:
        norm = np.mean((y_true[peaks] - y_pred[peaks]) ** 2, axis=-1)
    else:
        raise ValueError("p must be 1 or 2")
    return norm

def p3sw(y_pred: np.ndarray, y_true: np.ndarray, T: int = 3):
    """
    Implements the following formula:
    P3_{sw}(y, \hat{y}) = \sum_{t \in P} \big(y(t) - \max_{t' \in [t - T, t + T]} \hat{y}(t') \big)^2
    where P is the set of peaks in y
    """
    peaks = peaks_mask(y_true)
    y_true_peaks = y_true[peaks]

    # Calculate max(y_pred) in range [t - T, t + T]
    y_pred_padded = np.pad(y_pred, pad_width=((0, 0), (T, T)))
    y_pred_windows = np.stack([y_pred_padded[:, i:i + 2 * T + 1] for i in range(y_true.shape[1])], axis=1)
    y_pred_max_windows = np.max(y_pred_windows, axis=-1)

    pshift1 = np.mean((y_true_peaks - y_pred_max_windows[peaks]) ** 2, axis=-1)
    return pshift1

def p3eu(y_pred: np.ndarray, y_true: np.ndarray, alpha: float = 0.2, beta: float = 1.0, T: int = 10):
    peaks = peaks_mask(y_true)
    indices = np.arange(y_true.shape[1]) * np.sqrt(alpha)
    indices = np.vstack([indices] * y_true.shape[0])
    y_true *= np.sqrt(beta)
    y_pred *= np.sqrt(beta)
    y_true = np.stack([y_true, indices], axis=-1)
    y_pred = np.stack([y_pred, indices], axis=-1)
    y_true_peaks = y_true[peaks]

    y_pred_padded = np.pad(y_pred, pad_width=((0, 0), (T, T), (0, 0)))
    y_pred_windows = np.stack([y_pred_padded[:, i:i + 2 * T + 1] for i in range(y_true.shape[1])], axis=1)
    y_pred_peak_windows = y_pred_windows[peaks]
    distances = np.sum((y_pred_peak_windows - y_true_peaks[:, None]) ** 2, axis=-1)
    distances = np.min(distances, axis=-1)
    pshift2 = np.mean(distances, axis=-1)
    return pshift2

def all_peak_metrics(yp: np.ndarray, yt: np.ndarray):
    """
    yp.shape = yt.shape = (n_samples, n_timesteps)
    """
    metrics = {
        "mse": MSE,
        "mae": MAE,
        "pmse": lambda yt, yp: (pnorm(yt, yp, p=2) + pnorm(yp, yt, p=2))/2,
        "pmae": lambda yt, yp: (pnorm(yt, yp, p=1) + pnorm(yp, yt, p=1))/2,
        "p3sw": lambda yt, yp: (p3sw(yt, yp) + p3sw(yp, yt))/2,
        "p3eu": lambda yt, yp: (p3eu(yt, yp) + p3eu(yp, yt))/2,
    }
    return {k: v(yt, yp) for k, v in metrics.items()}

