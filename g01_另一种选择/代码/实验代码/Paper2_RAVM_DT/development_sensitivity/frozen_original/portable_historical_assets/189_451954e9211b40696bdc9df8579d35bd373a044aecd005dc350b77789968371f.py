"""D3d-1 pure-numpy *reference math*, NOT a full training runner.

Only allowed to process fully matured Update-role raw labels prior to candidate
adaptation. No future Service or hidden corruption mask is referenced.
"""
from __future__ import annotations
import numpy as np


def label_weights_from_update_raw(
    raw_y: np.ndarray,
    *,
    half_window: int = 5,
    additive_scale: float = 0.01,
    transition: float = 3.0,
    floor: float = 0.2,
) -> np.ndarray:
    """Soft down-weight outlying target labels using *within Update* raw labels.

    The entire Update segment must already have matured before this is called.
    Bidirectional neighbors refer exclusively to Update, not upcoming Service.
    """
    y = np.asarray(raw_y, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 * half_window + 1:
        raise ValueError('expected 1D matured Update labels, sufficient length')
    if not np.all(np.isfinite(y)):
        raise ValueError('non-finite matured Update labels')
    if half_window < 1 or not (0 < floor <= 1) or additive_scale <= 0 or transition <= 0:
        raise ValueError('invalid frozen parameters')
    # NumPy and PyTorch must not accidentally mix the roles of input and target.
    padded = np.pad(y, (half_window, half_window), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1)
    a = np.abs(y - med) / (mad + additive_scale)
    w = np.clip(np.minimum(1.0, transition / (a + 1e-8)), floor, 1.0)
    return w.astype(np.float32)


def target_weights_for_windows(raw_weights: np.ndarray, *, L: int = 96, H: int = 1) -> np.ndarray:
    """Map raw y weights onto each forecasting target from role-local windows().

    Exact frozen protocol mapping: window-row i -> raw target index L+i for H=1.
    """
    w = np.asarray(raw_weights)
    if w.ndim != 1 or H != 1 or L < 1 or len(w) - L - H + 1 <= 0:
        raise ValueError('unsupported target mapping; this study permits H=1 only')
    n = len(w) - L - H + 1
    out = w[L:L+n].copy()
    assert out.shape == (n,)
    return out


def weighted_mse_numpy(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Independent formula check; training runner must compute same sampled-batch loss in Torch."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(p) != len(y) or len(y) != len(w) or len(p) == 0:
        raise ValueError('loss input lengths differ')
    if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(w).all()) or np.any(w <= 0):
        raise ValueError('nonfinite / non-positive weighted loss input')
    return float(np.sum(w * np.square(p-y)) / np.sum(w))
