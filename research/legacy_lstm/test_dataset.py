"""Tests for the sample builder and the chronological (leak-free) split.

These exercise the pure numpy/pandas logic in ``dataset.py``. ``torch`` is not
required to run them: because ``dataset`` imports torch at module load only for
``StockDataset``/``DataLoader`` (which these tests do not touch), we install a
minimal torch stub before importing it. Run with ``python test_dataset.py`` or
``pytest test_dataset.py``.
"""

import sys
import types

import numpy as np
import pandas as pd


def _install_torch_stub():
    """Provide just enough of ``torch`` for ``import dataset`` to succeed."""
    if "torch" in sys.modules:
        return
    torch_stub = types.ModuleType("torch")
    torch_stub.from_numpy = lambda a: a
    torch_stub.tensor = lambda *a, **k: None
    torch_stub.float32 = "float32"

    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")

    class _Dataset:  # pragma: no cover - trivial stub
        pass

    class _DataLoader:  # pragma: no cover - trivial stub
        def __init__(self, *a, **k):
            pass

    data_mod.Dataset = _Dataset
    data_mod.DataLoader = _DataLoader
    utils_mod.data = data_mod
    torch_stub.utils = utils_mod

    sys.modules["torch"] = torch_stub
    sys.modules["torch.utils"] = utils_mod
    sys.modules["torch.utils.data"] = data_mod


def _install_tqdm_stub():
    """Provide a no-op ``tqdm`` so ``import dataset`` succeeds without the dep."""
    if "tqdm" in sys.modules:
        return
    tqdm_mod = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *a, **k):
        return iterable if iterable is not None else []

    _tqdm.write = lambda *a, **k: None
    tqdm_mod.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_mod


_install_torch_stub()
_install_tqdm_stub()

from dataset import build_samples, chronological_split  # noqa: E402


def _make_df(prices):
    n = len(prices)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "close": np.asarray(prices, dtype=np.float32),
        "volume": np.linspace(1000, 2000, n).astype(np.float32),
    })


def test_build_samples_shapes_and_finiteness():
    prices = 100 * (1.003 ** np.arange(200))  # gentle uptrend
    samples = build_samples(_make_df(prices), X=40, Y=20)

    assert len(samples) > 0, "expected at least one sample"
    for x, label in samples:
        assert x.shape == (40, 7), f"bad feature shape {x.shape}"
        assert np.isfinite(x).all(), "features must be finite (NaN check)"
        assert label in (0, 1), f"label must be binary, got {label}"


def test_labels_track_direction():
    # Strong uptrend: +12% take-profit is reached within the 20-day horizon.
    up = build_samples(_make_df(100 * (1.02 ** np.arange(160))), X=40, Y=20)
    up_labels = [y for _, y in up]
    assert sum(up_labels) / len(up_labels) > 0.9, "uptrend should be mostly positive"

    # Strong downtrend: -6% stop-loss is hit before any +12%, so all negative.
    down = build_samples(_make_df(100 * (0.98 ** np.arange(160))), X=40, Y=20)
    down_labels = [y for _, y in down]
    assert sum(down_labels) == 0, "downtrend must produce no positive labels"


def test_chronological_split_is_leak_free():
    X, Y = 40, 20
    purge = X + Y
    # Use index sentinels as opaque "samples"; the split only slices lists.
    n = 1000
    grouped = [[(i, 0) for i in range(n)]]

    train, val, test = chronological_split(
        grouped, train_frac=0.7, val_frac=0.15, purge=purge
    )

    tr = [i for i, _ in train]
    va = [i for i, _ in val]
    te = [i for i, _ in test]

    assert tr and va and te, "no split should be empty"

    # Strictly chronological: train precedes val precedes test.
    assert max(tr) < min(va) < max(va) < min(te)

    # Embargo gap prevents overlapping windows/label-horizons across boundaries.
    assert min(va) - max(tr) >= purge, "train/val gap smaller than purge"
    assert min(te) - max(va) >= purge, "val/test gap smaller than purge"

    # No index appears in more than one split.
    assert not (set(tr) & set(va)), "train/val overlap"
    assert not (set(va) & set(te)), "val/test overlap"
    assert not (set(tr) & set(te)), "train/test overlap"


def test_chronological_split_rejects_insufficient_history():
    # Too few samples to carve a val/test set after the purge gap.
    grouped = [[(i, 0) for i in range(50)]]
    try:
        chronological_split(grouped, purge=60)
    except ValueError:
        return
    raise AssertionError("expected ValueError on insufficient history")


def _run_all():
    tests = [
        test_build_samples_shapes_and_finiteness,
        test_labels_track_direction,
        test_chronological_split_is_leak_free,
        test_chronological_split_rejects_insufficient_history,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
