import cupy as cp


_ONEHOT_EYE_CACHE: dict[tuple[int, int, str], cp.ndarray] = {}
_WINDOW_INDEX_CACHE: dict[tuple[int, int, int], cp.ndarray] = {}


def _cached_eye(V: int, dtype) -> cp.ndarray:
    device_id = cp.cuda.Device().id
    dtype_name = cp.dtype(dtype).str
    cache_key = (device_id, int(V), dtype_name)
    eye = _ONEHOT_EYE_CACHE.get(cache_key)
    if eye is None:
        # Reuse one-hot basis per device/dtype to avoid repeated allocations.
        eye = cp.eye(V, dtype=dtype)
        _ONEHOT_EYE_CACHE[cache_key] = eye
    return eye


def _cached_window_ids(T: int, K: int) -> cp.ndarray:
    device_id = cp.cuda.Device().id
    cache_key = (device_id, int(T), int(K))
    window_ids = _WINDOW_INDEX_CACHE.get(cache_key)
    if window_ids is None:
        # Precompute [T-K, K] sliding indices once per shape.
        starts = cp.arange(T - K, dtype=cp.int32)[:, None]
        offsets = cp.arange(K, dtype=cp.int32)[None, :]
        window_ids = starts + offsets
        _WINDOW_INDEX_CACHE[cache_key] = window_ids
    return window_ids

def make_windows(tokens: cp.ndarray, K: int) -> tuple[cp.ndarray, cp.ndarray]:
    assert tokens.ndim == 2, f"tokens must be 2D, got {tokens.ndim}D"
    B, T = tokens.shape
    assert 0 < K < T, f"Need 0 < K < T, got K={K}, T={T}"

    window_ids = _cached_window_ids(T, K)

    # Flatten batch/time windows so each row is one supervised example.
    ctx = tokens[:, window_ids].reshape(B * (T - K), K)
    y = tokens[:, K:].reshape(B * (T - K),)
    return ctx.astype(cp.int32, copy=False), y.astype(cp.int32, copy=False)

def ctx_to_onehot_concat(ctx_tokens: cp.ndarray, V: int, dtype) -> cp.ndarray:
    assert ctx_tokens.ndim == 2, f"ctx_tokens must be 2D, got {ctx_tokens.ndim}D"
    eye = _cached_eye(V, dtype)
    onehot = eye[ctx_tokens.astype(cp.int32, copy=False)]
    # Concatenate K one-hot vectors into a single feature row.
    return onehot.reshape(ctx_tokens.shape[0], ctx_tokens.shape[1] * V)

def stable_softmax(logits: cp.ndarray, axis: int = 1) -> cp.ndarray:
    logits = logits.astype(cp.float32, copy=False)
    m = cp.max(logits, axis=axis, keepdims=True)
    ex = cp.exp(logits - m)
    s = cp.sum(ex, axis=axis, keepdims=True)
    return ex / s

def sample_categorical(rng, probs: cp.ndarray) -> cp.ndarray:
    assert probs.ndim == 2, f"probs must be 2D, got {probs.ndim}D"
    cdf = cp.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    r = rng.random((probs.shape[0], 1)).astype(probs.dtype, copy=False)
    return cp.argmax(cdf >= r, axis=1).astype(cp.int32, copy=False)

def cross_entropy_from_logits(logits: cp.ndarray, targets: cp.ndarray) -> cp.ndarray:
    logits = logits.astype(cp.float32, copy=False)
    targets = targets.astype(cp.int32, copy=False)
    row_ids = cp.arange(logits.shape[0], dtype=cp.int32)
    m = cp.max(logits, axis=1, keepdims=True)
    shifted = logits - m
    logsumexp = cp.log(cp.sum(cp.exp(shifted), axis=1)) + m[:, 0]
    target_logits = logits[row_ids, targets]
    return cp.mean(logsumexp - target_logits)

def accuracy_from_logits(logits: cp.ndarray, targets: cp.ndarray) -> cp.ndarray:
    preds = cp.argmax(logits, axis=1)
    return cp.mean((preds == targets.astype(cp.int32, copy=False)).astype(cp.float32))