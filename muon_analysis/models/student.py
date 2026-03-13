import cupy as cp

from muon_analysis.models.mlp_ar import MLP_AR
from muon_analysis.models.model_utils import accuracy_from_logits, cross_entropy_from_logits, ctx_to_onehot_concat, make_windows


class Student(MLP_AR):
	def loss_on_tokens(self, tokens):
		ctx, y = make_windows(tokens, self.K)
		u = ctx_to_onehot_concat(ctx, self.V, self.dtype)
		logits = self.forward_logits(u)
		return cross_entropy_from_logits(logits, y)

	def metrics_on_tokens(self, tokens):
		ctx, y = make_windows(tokens, self.K)
		u = ctx_to_onehot_concat(ctx, self.V, self.dtype)
		logits = self.forward_logits(u)
		ce = cross_entropy_from_logits(logits, y)
		acc = accuracy_from_logits(logits, y)
		return {
			"ce": float(ce.get()),
			"acc": float(acc.get()),
		}
