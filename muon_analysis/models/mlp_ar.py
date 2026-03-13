import cupy as cp
from math import sqrt

# Parent model class.
class MLP_AR:
    def __init__(
        self,
        V: int, # Vocab Size
        K: int, # Context Length
        width: int, # hidden width m
        layers: int, # Num of layers (L=1 means W1 -> Wout)
        *, # After this point args must be explicity declared
        activation: str = "relu", # relu or gelu
        dtype=cp.float32, # the default but open for future experimentation
        seed: int | None = None, # To initialize weights deterministically
        params: dict | None = None # Optional preloaded parameters dictionary
    ):
        assert V > 0 and K > 0 and width > 0 and layers >= 1, "Parameters are invalid."
        assert activation in ("relu", "gelu"), "Activation unsupported."

        self.V = int(V)
        self.K = int(K)
        self.m = int(width)
        self.L = int(layers)
        self.activation_name = activation
        self.dtype = dtype

        self.in_dim = self.K * self.V  # K*V

        if params is not None:
            self._load_params(params)
        else:
            if seed is None:
                seed = 0
            self._init_params(seed)

    @staticmethod
    def _xavier_uniform(rng, shape, fan_in, fan_out, dtype):
        a = sqrt(6.0 / (fan_in + fan_out)) # only needed once so no need for gpu
        return rng.uniform(-a, a, size=shape).astype(dtype, copy=False)

    def _init_params(self, seed: int):
        rng = cp.random.default_rng(int(seed))

        Ws = []
        #W1
        Ws.append(self._xavier_uniform(rng, (self.in_dim, self.m), self.in_dim, self.m, self.dtype))
        # W2...WL (if L>1)
        for _ in range(self.L - 1):
            Ws.append(self._xavier_uniform(rng, (self.m, self.m), self.m, self.m, self.dtype))

        Wout = self._xavier_uniform(rng, (self.m, self.V), self.m, self.V, self.dtype)

        self.Ws = Ws
        self.Wout = Wout

    def state_dict(self) -> dict:
        sd = {f"W{i+1}": W for i, W in enumerate(self.Ws)} # setting numbered layer weight keys
        sd["Wout"] = self.Wout
        return sd # returns a dict of the parameters for saving

    def copy_state_dict(self) -> dict:
        return {key: cp.array(value, copy=True) for key, value in self.state_dict().items()}

    def load_state_dict(self, params: dict):
        self._load_params(params)

    # Loads weights and checks that weights exist/align with dimensions properly
    def _load_params(self, params: dict):
        Ws = []
        for i in range(self.L):
            key = f"W{i+1}"
            assert key in params, f"Missing {key} in params!"
            W = cp.asarray(params[key], dtype=self.dtype)
            Ws.append(W)
        
        assert "Wout" in params, "Missing Wout in params!"
        Wout = cp.asarray(params["Wout"], dtype=self.dtype)

        # Check that shapes/dims of weights valid.
        assert Ws[0].shape == (self.in_dim, self.m), f"W1 shape {Ws[0].shape} NEQ {(self.in_dim, self.m)}!"
        for i in range(1, self.L):
                assert Ws[i].shape == (self.m, self.m), f"W{i+1} shape {Ws[i].shape} NEQ {(self.m, self.m)}!"
        assert Wout.shape == (self.m, self.V), f"Wout shape {Wout.shape} NEQ {(self.m, self.V)}!"

        self.Ws = Ws
        self.Wout = Wout

    def _activation(self, x):
        if self.activation_name == "relu":
            return cp.maximum(x,0)
        # GELU approximate taken from PyTorch
        c = cp.sqrt(2.0 / cp.pi).astype(x.dtype)
        return 0.5 * x * (1.0 + cp.tanh(c * (x + 0.044715 * x * x * x)))

    # Compute logits from feature input u
    # u: (N, K*V) float tensor
    # -> logits: (N, V) float tensor
    def forward_logits(self, u: cp.ndarray) -> cp.ndarray:
        assert u.ndim == 2, f"u must be 2D, got {u.ndim}D"
        assert u.shape[1] == self.in_dim, f"u second dim {u.shape[1]} NEQ in_dim {self.in_dim}!"

        h = u.astype(self.dtype, copy=False) # h: (N, K*V)
        for W in self.Ws:
            h = self._activation(h @ W) # after h: (N, m), hidden activation matrix
        logits = h @ self.Wout
        return logits

    def parameter_shapes(self) -> dict:
        return {key: value.shape for key, value in self.state_dict().items()}

    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.state_dict().values()))

