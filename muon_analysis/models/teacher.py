import cupy as cp

from muon_analysis.models.mlp_ar import MLP_AR
from muon_analysis.models.model_utils import ctx_to_onehot_concat, sample_categorical, stable_softmax

class Teacher(MLP_AR):
    def sample_next(self, ctx_tokens, rng, temperature=None):
        assert ctx_tokens.ndim == 2, f"ctx_tokens must be 2D, got {ctx_tokens.ndim}D"
        temp = float(temperature if temperature is not None else 1.0)
        assert temp > 0.0, f"temperature must be positive, got {temp}"

        u = ctx_to_onehot_concat(ctx_tokens, self.V, self.dtype)
        logits = self.forward_logits(u)
        # Temperature scaling before categorical sampling.
        probs = stable_softmax(logits / temp, axis=1)
        return sample_categorical(rng, probs)

    def generate_sequences(self, B, T, seed=None, prefix_mode="bos", bos_id=0, temperature=None):
        assert T > self.K, f"Need T > K, got T={T}, K={self.K}"
        assert prefix_mode in ("bos", "random"), f"prefix_mode {prefix_mode} unsupported"

        rng = cp.random.default_rng(seed)
        tokens = cp.empty((int(B), int(T)), dtype=cp.int32)

        if prefix_mode == "bos":
            tokens[:, :self.K] = int(bos_id)
        else:
            tokens[:, :self.K] = rng.integers(0, self.V, size=(int(B), self.K), dtype=cp.int32)

        # Autoregressive rollout from left to right.
        for t in range(self.K, int(T)):
            ctx_tokens = tokens[:, t - self.K:t]
            tokens[:, t] = self.sample_next(ctx_tokens, rng, temperature=temperature)

        return tokens
