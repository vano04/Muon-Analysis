import json
import os
import tempfile
from pathlib import Path

import cupy as cp
import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def _numpy_array_to_cupy(array):
    # Restore bf16 arrays saved through NumPy's raw-void representation.
    if array.dtype.kind == "V" and array.dtype.itemsize == 2 and ml_dtypes is not None:
        array = array.view(ml_dtypes.bfloat16)
    return cp.asarray(array)


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(fd)

    try:
        written_path = writer(tmp_path)
        # Move into place atomically to avoid partial files on interruption.
        os.replace(written_path or tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_npz(path, dict_of_arrays):
    def _writer(tmp_path):
        cp.savez(tmp_path, **dict_of_arrays)
        return tmp_path

    atomic_write(path, _writer)


def load_npz(path) -> dict:
    archive = np.load(path)
    try:
        if hasattr(archive, "files"):
            keys = archive.files
            getter = archive.__getitem__
        elif hasattr(archive, "npz_file"):
            keys = archive.npz_file.files
            getter = archive.__getitem__
        else:
            raise TypeError(f"Unsupported NPZ archive type: {type(archive)}")

        return {key: _numpy_array_to_cupy(getter(key)) for key in keys}
    finally:
        if hasattr(archive, "close"):
            archive.close()


def save_tokens(path, tokens):
    def _writer(tmp_path):
        cp.save(tmp_path, cp.asarray(tokens))
        return tmp_path

    atomic_write(path, _writer)


def load_tokens(path) -> cp.ndarray:
    return cp.asarray(cp.load(path))


def save_json(path, obj):
    def _writer(tmp_path):
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)

    atomic_write(path, _writer)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)