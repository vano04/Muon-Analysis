import cupy as cp
from math import sqrt


def _default_low_precision_dtype():
    try:
        return cp.dtype("bfloat16")
    except Exception:
        return cp.float16

def _float32_view(array):
    if array.dtype == cp.float32:
        return array
    return array.astype(cp.float32, copy=False)


def newton_schulz(G, steps=3, eps=1e-7, state_dtype=None):
    a, b, c = 3.4445, -4.7750, 2.0315

    if state_dtype is None:
        state_dtype = _default_low_precision_dtype()
    state_dtype = cp.dtype(state_dtype)

    Xf = _float32_view(G)
    # Normalize first for stable Newton-Schulz iterations.
    Xf = Xf / (cp.linalg.norm(Xf) + eps)

    t = Xf.shape[0] > Xf.shape[1]
    if t:
        Xf = Xf.T

    if state_dtype == cp.float32:
        X = Xf
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
        return X.T if t else X

    X = Xf.astype(state_dtype, copy=False)

    for _ in range(steps):
        Xf = X.astype(cp.float32, copy=False)
        A = Xf @ Xf.T
        X = (a * Xf + (b * A + c * (A @ A)) @ Xf).astype(state_dtype, copy=False)

    return X.T if t else X


def _muon_update_scale(shape):
    if len(shape) < 2:
        return 1.0
    rows, cols = shape[0], shape[1]
    return sqrt(max(1.0, rows / cols))


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.M = [cp.zeros_like(param, dtype=cp.float32) for param in params]
        self.V = [cp.zeros_like(param, dtype=cp.float32) for param in params]

    def step(self, params, grads):
        self.t += 1
        bias_correction1 = 1.0 - self.beta1 ** self.t
        bias_correction2_sqrt = sqrt(1.0 - self.beta2 ** self.t)

        for idx, (param, grad) in enumerate(zip(params, grads)):
            grad32 = grad.astype(cp.float32, copy=False)
            self.M[idx] *= self.beta1
            self.M[idx] += (1.0 - self.beta1) * grad32
            self.V[idx] *= self.beta2
            self.V[idx] += (1.0 - self.beta2) * (grad32 * grad32)

            update = cp.sqrt(self.V[idx])
            update /= bias_correction2_sqrt
            update += self.eps
            cp.divide(self.M[idx], update, out=update)
            update /= bias_correction1

            if self.weight_decay != 0.0:
                param *= (1.0 - self.lr * self.weight_decay)
            param -= (self.lr * update).astype(param.dtype, copy=False)

        return params


class Muon:
    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.9, ns_steps=3, eps=1e-7, ns_state_dtype=None, nesterov=True):
        self.lr = lr
        self.weight_decay = weight_decay
        self.mu = momentum
        self.ns_steps = ns_steps
        self.eps = eps
        self.ns_state_dtype = ns_state_dtype
        self.nesterov = nesterov
        self.B = [cp.zeros_like(param, dtype=cp.float32) for param in params]

    def step(self, params, grads):
        for idx, (param, grad) in enumerate(zip(params, grads)):
            grad32 = _float32_view(grad)
            momentum_buffer = self.B[idx]
            momentum_buffer *= self.mu
            momentum_buffer += grad32
            update_source = grad32 + self.mu * momentum_buffer if self.nesterov else momentum_buffer
            update = newton_schulz(update_source, steps=self.ns_steps, eps=self.eps, state_dtype=self.ns_state_dtype)
            step_scale = cp.float32(self.lr * _muon_update_scale(param.shape))

            if self.weight_decay != 0.0:
                param *= (1.0 - self.lr * self.weight_decay)
            scaled_update = update * step_scale
            if scaled_update.dtype != param.dtype:
                scaled_update = scaled_update.astype(param.dtype, copy=False)
            param -= scaled_update

        return params


class MuonHybrid:
    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.9, ns_steps=3, eps=1e-7, ns_state_dtype=None, nesterov=True):
        # Muon for matrix-like params, AdamW for vectors/scalars.
        self._muon_indices = [idx for idx, param in enumerate(params) if param.ndim >= 2]
        self._adamw_indices = [idx for idx, param in enumerate(params) if param.ndim < 2]
        self._muon = None
        self._adamw = None

        if self._muon_indices:
            muon_params = [params[idx] for idx in self._muon_indices]
            self._muon = Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                ns_steps=ns_steps,
                eps=eps,
                ns_state_dtype=ns_state_dtype,
                nesterov=nesterov,
            )

        if self._adamw_indices:
            adamw_params = [params[idx] for idx in self._adamw_indices]
            self._adamw = AdamW(adamw_params, lr=lr, weight_decay=weight_decay)

    def step(self, params, grads):
        if self._muon is not None:
            muon_params = [params[idx] for idx in self._muon_indices]
            muon_grads = [grads[idx] for idx in self._muon_indices]
            updated = self._muon.step(muon_params, muon_grads)
            for offset, idx in enumerate(self._muon_indices):
                params[idx] = updated[offset]

        if self._adamw is not None:
            adamw_params = [params[idx] for idx in self._adamw_indices]
            adamw_grads = [grads[idx] for idx in self._adamw_indices]
            updated = self._adamw.step(adamw_params, adamw_grads)
            for offset, idx in enumerate(self._adamw_indices):
                params[idx] = updated[offset]

        return params

def build_optimizer(name, params, lr, weight_decay, muon_ns_steps=3):
    # Small factory used by training and benchmark scripts.
    name = str(name).lower()
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "muon":
        return Muon(params, lr=lr, weight_decay=weight_decay, ns_steps=muon_ns_steps)
    if name == "muon_hybrid":
        return MuonHybrid(params, lr=lr, weight_decay=weight_decay, ns_steps=muon_ns_steps)
    raise ValueError(f"Unsupported optimizer {name}")