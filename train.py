import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from time import perf_counter

import cupy as cp
import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muon_analysis.config import Config
from muon_analysis.dtypes import resolve_dtype
from muon_analysis.io_utils import load_npz, load_tokens, save_json, save_npz
from muon_analysis.models.model_utils import accuracy_from_logits, cross_entropy_from_logits, ctx_to_onehot_concat, make_windows, stable_softmax
from muon_analysis.models.student import Student
from muon_analysis.models.teacher import Teacher
from muon_analysis.optim import build_optimizer


def _state_to_params(student: Student):
	return [*student.Ws, student.Wout]


def _apply_param_list(student: Student, params):
	student.Ws = [param for param in params[:-1]]
	student.Wout = params[-1]


def _activation_and_grad(name: str, x: cp.ndarray):
	if name == "relu":
		y = cp.maximum(x, 0)
		dy = (x > 0).astype(x.dtype)
		return y, dy

	c = cp.sqrt(cp.asarray(2.0 / cp.pi, dtype=x.dtype))
	x3 = x * x * x
	inner = c * (x + 0.044715 * x3)
	tanh_inner = cp.tanh(inner)
	sech2 = 1.0 - tanh_inner * tanh_inner
	y = 0.5 * x * (1.0 + tanh_inner)
	dy = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
	return y, dy


def _log_softmax(logits: cp.ndarray, axis: int = 1) -> cp.ndarray:
	logits32 = logits.astype(cp.float32, copy=False)
	m = cp.max(logits32, axis=axis, keepdims=True)
	shifted = logits32 - m
	return shifted - cp.log(cp.sum(cp.exp(shifted), axis=axis, keepdims=True))


def _run_models(student: Student, teacher: Teacher, tokens: cp.ndarray, temperature: float):
	ctx, targets = make_windows(tokens, student.K)
	u = ctx_to_onehot_concat(ctx, student.V, student.dtype)
	teacher_logits = teacher.forward_logits(u)
	student_logits = student.forward_logits(u)
	teacher_scaled = teacher_logits.astype(cp.float32, copy=False) / temperature
	teacher_probs = stable_softmax(teacher_scaled, axis=1)
	return {
		"u": u,
		"targets": targets,
		"student_logits": student_logits,
		"teacher_probs": teacher_probs,
	}


def _collect_train_metrics(student_logits: cp.ndarray, teacher_probs: cp.ndarray, targets: cp.ndarray, temperature: float) -> dict:
	student_scaled = student_logits.astype(cp.float32, copy=False) / temperature
	student_log_probs = _log_softmax(student_scaled, axis=1)
	distill_ce = -cp.mean(cp.sum(teacher_probs * student_log_probs, axis=1))
	teacher_acc = cp.mean((cp.argmax(student_logits, axis=1) == cp.argmax(teacher_probs, axis=1)).astype(cp.float32))
	token_ce = cross_entropy_from_logits(student_logits, targets)
	token_acc = accuracy_from_logits(student_logits, targets)
	metric_values = cp.asarray([distill_ce, teacher_acc, token_ce, token_acc], dtype=cp.float32).get()
	return {
		"distill_ce": float(metric_values[0]),
		"teacher_acc": float(metric_values[1]),
		"token_ce": float(metric_values[2]),
		"token_acc": float(metric_values[3]),
	}


def _forward_backward(
	student: Student,
	teacher: Teacher,
	tokens: cp.ndarray,
	temperature: float,
	compute_grad_norm: bool = False,
	compute_metrics: bool = False,
):
	model_outputs = _run_models(student, teacher, tokens, temperature)
	u = model_outputs["u"]
	teacher_probs = model_outputs["teacher_probs"]
	logits = model_outputs["student_logits"]
	targets = model_outputs["targets"]

	hidden_inputs = [u]
	act_grads = []
	h = u.astype(student.dtype, copy=False)

	for W in student.Ws:
		pre = h @ W
		h, act_grad = _activation_and_grad(student.activation_name, pre)
		hidden_inputs.append(h)
		act_grads.append(act_grad)

	student_scaled = logits.astype(cp.float32, copy=False) / temperature
	student_probs = stable_softmax(student_scaled, axis=1)
	grad_logits = (student_probs - teacher_probs) / (logits.shape[0] * temperature)

	grads = [None] * (len(student.Ws) + 1)
	grads[-1] = hidden_inputs[-1].T @ grad_logits
	grad_h = grad_logits @ student.Wout.T

	for layer_id in range(len(student.Ws) - 1, -1, -1):
		grad_pre = grad_h * act_grads[layer_id]
		grads[layer_id] = hidden_inputs[layer_id].T @ grad_pre
		grad_h = grad_pre @ student.Ws[layer_id].T

	grad_norm = None
	if compute_grad_norm:
		grad_norm_sq = cp.zeros((), dtype=cp.float32)
		for grad in grads:
			grad32 = grad.astype(cp.float32, copy=False)
			grad_norm_sq += cp.sum(grad32 * grad32)
		grad_norm = float(cp.sqrt(grad_norm_sq).get())

	metrics = {
		"distill_ce": None,
		"teacher_acc": None,
		"token_ce": None,
		"token_acc": None,
	}
	if compute_metrics:
		metrics = _collect_train_metrics(logits, teacher_probs, targets, temperature)

	return grads, {
		"distill_ce": metrics["distill_ce"],
		"teacher_acc": metrics["teacher_acc"],
		"token_ce": metrics["token_ce"],
		"token_acc": metrics["token_acc"],
		"grad_norm": grad_norm,
	}


def evaluate_student(student: Student, teacher: Teacher, tokens: cp.ndarray, temperature: float) -> dict:
	model_outputs = _run_models(student, teacher, tokens, temperature)
	return _collect_train_metrics(
		model_outputs["student_logits"],
		model_outputs["teacher_probs"],
		model_outputs["targets"],
		temperature,
	)


def _param_norm(student: Student) -> float:
	norm_sq = cp.zeros((), dtype=cp.float32)
	for param in _state_to_params(student):
		param32 = param.astype(cp.float32, copy=False)
		norm_sq += cp.sum(param32 * param32)
	return float(cp.sqrt(norm_sq).get())


def _format_csv_metric(value) -> str:
	return "" if value is None else f"{value:.6f}"


def _generate_train_tokens(teacher: Teacher, config: Config, step: int):
	tokens = teacher.generate_sequences(
		config.B_train,
		config.T,
		seed=config.train_seed + step,
		prefix_mode=config.prefix_mode,
		bos_id=config.bos_id,
		temperature=config.temperature,
	)
	cp.cuda.get_current_stream().synchronize()
	return tokens


def _generate_train_block(teacher: Teacher, config: Config, start_step: int, block_steps: int):
	total_batch = int(config.B_train * block_steps)
	tokens = teacher.generate_sequences(
		total_batch,
		config.T,
		seed=config.train_seed + start_step,
		prefix_mode=config.prefix_mode,
		bos_id=config.bos_id,
		temperature=config.temperature,
	)
	cp.cuda.get_current_stream().synchronize()
	return tokens.reshape(block_steps, config.B_train, config.T)


def _train_tokens_path(artifact_dir: Path, train_seed: int) -> Path:
	return artifact_dir / "train_tokens" / f"seed_{int(train_seed)}.npy"


class _TrainBatchPrefetcher:
	def __init__(self, teacher: Teacher, config: Config):
		self.teacher = teacher
		self.config = config
		self.executor = None
		self.future = None
		self.step = None
		if config.num_threads > 1:
			self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-batch")

	def schedule(self, step: int):
		if step > self.config.steps:
			return
		if self.executor is None:
			self.step = step
			return
		self.future = self.executor.submit(_generate_train_tokens, self.teacher, self.config, step)
		self.step = step

	def get(self, step: int):
		if self.executor is None:
			return _generate_train_tokens(self.teacher, self.config, step)
		if self.future is None or self.step != step:
			self.schedule(step)
		tokens = self.future.result()
		self.future = None
		self.step = None
		return tokens

	def close(self):
		if self.executor is not None:
			self.executor.shutdown(wait=True, cancel_futures=True)


class _PrecomputedTrainBatchSource:
	def __init__(self, teacher: Teacher, config: Config):
		self.teacher = teacher
		self.config = config
		self.block_size = max(1, min(config.train_batch_block_size, config.steps))
		self.executor = None
		self.current_block = None
		self.current_start_step = None
		self.future = None
		self.future_start_step = None
		if config.num_threads > 1:
			self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train-block")

	def _block_step_count(self, start_step: int) -> int:
		return min(self.block_size, self.config.steps - start_step + 1)

	def _start_block(self, start_step: int):
		block_steps = self._block_step_count(start_step)
		if self.executor is None:
			return _generate_train_block(self.teacher, self.config, start_step, block_steps)
		self.future = self.executor.submit(_generate_train_block, self.teacher, self.config, start_step, block_steps)
		self.future_start_step = start_step
		return None

	def _schedule_next_block(self, start_step: int):
		if start_step > self.config.steps:
			return
		if self.future is None:
			self._start_block(start_step)

	def get(self, step: int):
		if self.current_block is None or self.current_start_step is None or not (self.current_start_step <= step < self.current_start_step + self.current_block.shape[0]):
			if self.future is not None and self.future_start_step == step:
				self.current_block = self.future.result()
				self.current_start_step = self.future_start_step
				self.future = None
				self.future_start_step = None
			else:
				self.current_block = self._start_block(step)
				self.current_start_step = step
				if self.current_block is None:
					self.current_block = self.future.result()
					self.current_start_step = self.future_start_step
					self.future = None
					self.future_start_step = None

			next_start_step = self.current_start_step + self.current_block.shape[0]
			self._schedule_next_block(next_start_step)

		local_index = step - self.current_start_step
		return self.current_block[local_index]

	def close(self):
		if self.executor is not None:
			self.executor.shutdown(wait=True, cancel_futures=True)


class _DiskTrainBatchSource:
	def __init__(self, tokens_path: Path, config: Config):
		self.config = config
		self.tokens_path = tokens_path
		self.block_size = max(1, min(config.train_batch_block_size, config.steps))
		self.train_tokens = np.load(tokens_path, mmap_mode="r")
		expected_shape = (int(config.steps), int(config.B_train), int(config.T))
		if tuple(self.train_tokens.shape) != expected_shape:
			raise ValueError(f"Train token shape mismatch for {tokens_path}: {self.train_tokens.shape} != {expected_shape}")
		self.current_block = None
		self.current_start_step = None

	def _load_block(self, start_step: int):
		block_steps = min(self.block_size, self.config.steps - start_step + 1)
		cpu_block = self.train_tokens[start_step - 1:start_step - 1 + block_steps]
		self.current_block = cp.asarray(cpu_block)
		self.current_start_step = start_step

	def get(self, step: int):
		if self.current_block is None or self.current_start_step is None or not (self.current_start_step <= step < self.current_start_step + self.current_block.shape[0]):
			self._load_block(step)

		local_index = step - self.current_start_step
		return self.current_block[local_index]

	def close(self):
		self.current_block = None
		self.current_start_step = None
		self.train_tokens = None


def _make_train_batch_source(teacher: Teacher, config: Config, artifact_dir: Path):
	tokens_path = _train_tokens_path(artifact_dir, config.train_seed)
	if tokens_path.exists():
		return _DiskTrainBatchSource(tokens_path, config)
	if config.train_data_mode == "precomputed":
		return _PrecomputedTrainBatchSource(teacher, config)
	return _TrainBatchPrefetcher(teacher, config)


def _save_metrics_csv(path: Path, rows: list[dict]):
	with path.open("w", newline="", encoding="utf-8") as handle:
		fieldnames = [
			"step",
			"elapsed_sec",
			"iter_per_sec",
			"tokens_per_sec",
			"train_distill_ce",
			"train_teacher_acc",
			"train_token_ce",
			"train_token_acc",
			"grad_norm",
			"param_norm",
			"val_distill_ce",
			"val_teacher_acc",
			"val_token_ce",
			"val_token_acc",
		]
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _make_step_log_writer(path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	handle = path.open("w", newline="", encoding="utf-8")
	fieldnames = [
		"step",
		"elapsed_sec",
		"step_time_sec",
		"iter_per_sec",
		"tokens_per_sec",
		"train_distill_ce",
		"train_teacher_acc",
		"train_token_ce",
		"train_token_acc",
		"grad_norm",
		"param_norm",
	]
	writer = csv.DictWriter(handle, fieldnames=fieldnames)
	writer.writeheader()
	handle.flush()
	return handle, writer


def _make_eval_log_writer(path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	handle = path.open("w", newline="", encoding="utf-8")
	fieldnames = [
		"step",
		"elapsed_sec",
		"iter_per_sec",
		"tokens_per_sec",
		"val_distill_ce",
		"val_teacher_acc",
		"val_token_ce",
		"val_token_acc",
	]
	writer = csv.DictWriter(handle, fieldnames=fieldnames)
	writer.writeheader()
	handle.flush()
	return handle, writer


def _should_print_progress(step: int, total_steps: int, log_every: int, evaluated: bool) -> bool:
	return step == 1 or step == total_steps or step % log_every == 0 or evaluated


def _print_progress(config: Config, step: int, elapsed_sec: float, iter_per_sec: float, train_metrics: dict, param_norm: float, val_metrics: dict | None):
	message = (
		f"[{config.optimizer}] {config.tier_name} step {step}/{config.steps} "
		f"elapsed={elapsed_sec:.1f}s it/s={iter_per_sec:.2f} "
		f"train_distill_ce={train_metrics['distill_ce']:.6f} train_teacher_acc={train_metrics['teacher_acc']:.6f} "
		f"grad_norm={train_metrics['grad_norm']:.6f} param_norm={param_norm:.6f}"
	)
	if val_metrics is not None:
		message += (
			f" val_distill_ce={val_metrics['distill_ce']:.6f} val_teacher_acc={val_metrics['teacher_acc']:.6f} "
			f"val_token_ce={val_metrics['token_ce']:.6f} val_token_acc={val_metrics['token_acc']:.6f}"
		)
	print(message, flush=True)


def _plot_metric(run_dir: Path, rows: list[dict], train_key: str, val_key: str, ylabel: str, filename: str):
	plt.figure(figsize=(9, 5))
	steps = [row["step"] for row in rows]
	train_values = [row[train_key] for row in rows]
	val_values = [row[val_key] for row in rows]
	plt.plot(steps, train_values, linewidth=1.8, label=f"train {ylabel.lower()}")
	plt.plot(steps, val_values, linewidth=2.2, linestyle="--", label=f"validation {ylabel.lower()}")
	plt.xlabel("Step")
	plt.ylabel(ylabel)
	plt.title(f"{ylabel} vs Step")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(run_dir / filename, dpi=160)
	plt.close()


def _plot_time_metric(run_dir: Path, rows: list[dict], filename: str):
	plt.figure(figsize=(9, 5))
	elapsed = [row["elapsed_sec"] for row in rows]
	values = [row["val_distill_ce"] for row in rows]
	plt.plot(elapsed, values, linewidth=2.0)
	plt.xlabel("Elapsed Seconds")
	plt.ylabel("Validation Distillation CE")
	plt.title("Validation Distillation CE vs Time")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(run_dir / filename, dpi=160)
	plt.close()


def _write_report(run_dir: Path, config: Config, teacher: Teacher, student: Student, summary: dict):
	random_ce = float(cp.log(cp.asarray(config.V, dtype=cp.float32)).get())
	random_acc = 1.0 / config.V
	lines = [
		f"# Trial {config.tier_name} {config.optimizer}",
		"",
		"## Config",
		"",
		f"- tier={config.tier_name}",
		f"- optimizer={config.optimizer}",
		f"- lr={config.lr}",
		f"- weight_decay={config.weight_decay}",
		f"- student_seed={config.student_seed}, train_seed={config.train_seed}",
		f"- width={config.width}, layers={config.layers}, V={config.V}, K={config.K}, T={config.T}",
		"",
		"## Baseline",
		"",
		f"- random CE baseline={random_ce:.6f}",
		f"- random accuracy baseline={random_acc:.6f}",
		"",
		"## Parameters",
		"",
		f"- teacher count={teacher.parameter_count()}",
		f"- student count={student.parameter_count()}",
		"",
		"## Summary",
		"",
		f"- status={summary['status']}",
		f"- steps_completed={summary['steps_completed']}",
		f"- best validation distill CE={summary['best_validation']['distill_ce']:.6f} at step {summary['best_validation']['step']}",
		f"- final validation distill CE={summary['final_validation']['distill_ce']:.6f}",
		f"- final test distill CE={summary['final_test']['distill_ce']:.6f}",
		"",
		"## Plots",
		"",
		"![Train vs Validation Distillation CE](plots_distill_ce.png)",
		"",
		"![Train vs Validation Accuracy](plots_accuracy.png)",
		"",
		"![Validation Distillation CE vs Time](plots_validation_time.png)",
	]
	(run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_analysis_artifacts(run_dir: Path, config: Config, teacher: Teacher, student: Student, summary: dict, rows: list[dict]):
	plot_jobs = [
		(_plot_metric, (run_dir, rows, "train_distill_ce", "val_distill_ce", "Distillation CE", "plots_distill_ce.png")),
		(_plot_metric, (run_dir, rows, "train_teacher_acc", "val_teacher_acc", "Teacher Agreement", "plots_accuracy.png")),
		(_plot_time_metric, (run_dir, rows, "plots_validation_time.png")),
	]
	plot_workers = min(len(plot_jobs), max(1, config.num_threads - 1))

	if plot_workers > 1:
		with ThreadPoolExecutor(max_workers=plot_workers, thread_name_prefix="analysis") as executor:
			futures = [executor.submit(job, *args) for job, args in plot_jobs]
			for future in futures:
				future.result()
	else:
		for job, args in plot_jobs:
			job(*args)

	_write_report(run_dir, config, teacher, student, summary)


def _all_finite(student: Student, grads: list[cp.ndarray]) -> bool:
	checks = [cp.isfinite(tensor).all() for tensor in [*_state_to_params(student), *grads]]
	if not checks:
		return True
	combined = checks[0]
	for check in checks[1:]:
		combined = cp.logical_and(combined, check)
	return bool(combined.get())


def train_run(
	config: Config,
	artifact_dir: Path,
	run_dir: Path | None = None,
	progress_desc: str | None = None,
	leave_progress: bool = False,
	show_progress: bool = True,
	progress_position: int = 0,
):
	run_dir = artifact_dir if run_dir is None else run_dir
	run_dir.mkdir(parents=True, exist_ok=True)

	dtype = resolve_dtype(config.dtype)
	teacher_state = load_npz(artifact_dir / "teacher_weights.npz")
	validation_tokens = load_tokens(artifact_dir / "validation_tokens.npy")
	test_tokens = load_tokens(artifact_dir / "test_tokens.npy")

	teacher = Teacher(
		config.V,
		config.K,
		config.width,
		config.layers,
		activation=config.activation,
		dtype=dtype,
		params=teacher_state,
	)
	student = Student(
		config.V,
		config.K,
		config.width,
		config.layers,
		activation=config.activation,
		dtype=dtype,
		seed=config.student_seed,
	)
	optimizer = build_optimizer(
		config.optimizer,
		_state_to_params(student),
		lr=config.lr,
		weight_decay=config.weight_decay,
		muon_ns_steps=config.muon_ns_steps,
	)

	rows = []
	log_every = config.log_every
	flush_every = config.step_log_flush_every
	flush_every_effective = 1 if not show_progress else flush_every
	step_log_handle, step_log_writer = _make_step_log_writer(run_dir / "step_log.csv")
	eval_log_handle, eval_log_writer = _make_eval_log_writer(run_dir / "eval_log.csv")
	batch_source = _make_train_batch_source(teacher, config, artifact_dir)
	status = "ok"
	non_finite_step = None
	steps_completed = 0
	start_time = perf_counter()
	last_step_time = start_time
	best_validation = None
	final_validation = None

	progress = tqdm(
		total=config.steps,
		desc=progress_desc or f"{config.optimizer}:{config.tier_name}",
		leave=leave_progress,
		disable=not show_progress,
		dynamic_ncols=True,
		position=progress_position,
	)
	progress_detail = tqdm(
		total=0,
		desc="",
		bar_format="{desc}",
		leave=leave_progress,
		disable=not show_progress,
		dynamic_ncols=True,
		position=progress_position + 1,
	)

	try:
		for step in range(1, config.steps + 1):
			evaluated = step % config.eval_every == 0 or step == 1 or step == config.steps
			printed = step == 1 or step == config.steps or step % log_every == 0
			finite_checked = step == 1 or evaluated or step % config.finite_check_every == 0
			collect_heavy_metrics = evaluated or printed or finite_checked

			tokens = batch_source.get(step)
			grads, train_metrics = _forward_backward(
				student,
				teacher,
				tokens,
				config.temperature,
				compute_grad_norm=collect_heavy_metrics,
				compute_metrics=collect_heavy_metrics,
			)

			if finite_checked and not _all_finite(student, grads):
				status = "diverged"
				non_finite_step = step
				break

			params = optimizer.step(_state_to_params(student), grads)
			_apply_param_list(student, params)

			if finite_checked and not _all_finite(student, []):
				status = "diverged"
				non_finite_step = step
				break

			now = perf_counter()
			elapsed_sec = now - start_time
			step_time_sec = now - last_step_time
			iter_per_sec = step / elapsed_sec if elapsed_sec > 0.0 else 0.0
			tokens_per_sec = (step * config.B_train * (config.T - config.K)) / elapsed_sec if elapsed_sec > 0.0 else 0.0
			last_step_time = now
			param_norm = _param_norm(student) if collect_heavy_metrics else None

			step_log_writer.writerow({
				"step": step,
				"elapsed_sec": f"{elapsed_sec:.6f}",
				"step_time_sec": f"{step_time_sec:.6f}",
				"iter_per_sec": f"{iter_per_sec:.6f}",
				"tokens_per_sec": f"{tokens_per_sec:.6f}",
				"train_distill_ce": _format_csv_metric(train_metrics["distill_ce"]),
				"train_teacher_acc": _format_csv_metric(train_metrics["teacher_acc"]),
				"train_token_ce": _format_csv_metric(train_metrics["token_ce"]),
				"train_token_acc": _format_csv_metric(train_metrics["token_acc"]),
				"grad_norm": _format_csv_metric(train_metrics["grad_norm"]),
				"param_norm": _format_csv_metric(param_norm),
			})
			if step % flush_every_effective == 0 or evaluated or printed:
				step_log_handle.flush()

			val_metrics = None
			if evaluated:
				val_metrics = evaluate_student(student, teacher, validation_tokens, config.temperature)
				row = {
					"step": step,
					"elapsed_sec": round(elapsed_sec, 6),
					"iter_per_sec": round(iter_per_sec, 6),
					"tokens_per_sec": round(tokens_per_sec, 6),
					"train_distill_ce": round(train_metrics["distill_ce"], 6),
					"train_teacher_acc": round(train_metrics["teacher_acc"], 6),
					"train_token_ce": round(train_metrics["token_ce"], 6),
					"train_token_acc": round(train_metrics["token_acc"], 6),
					"grad_norm": round(train_metrics["grad_norm"], 6) if train_metrics["grad_norm"] is not None else None,
					"param_norm": round(param_norm, 6) if param_norm is not None else None,
					"val_distill_ce": round(val_metrics["distill_ce"], 6),
					"val_teacher_acc": round(val_metrics["teacher_acc"], 6),
					"val_token_ce": round(val_metrics["token_ce"], 6),
					"val_token_acc": round(val_metrics["token_acc"], 6),
				}
				rows.append(row)
				final_validation = val_metrics
				if best_validation is None or val_metrics["distill_ce"] < best_validation["distill_ce"]:
					best_validation = {"step": step, **val_metrics}
				eval_log_writer.writerow({
					"step": step,
					"elapsed_sec": f"{elapsed_sec:.6f}",
					"iter_per_sec": f"{iter_per_sec:.6f}",
					"tokens_per_sec": f"{tokens_per_sec:.6f}",
					"val_distill_ce": f"{val_metrics['distill_ce']:.6f}",
					"val_teacher_acc": f"{val_metrics['teacher_acc']:.6f}",
					"val_token_ce": f"{val_metrics['token_ce']:.6f}",
					"val_token_acc": f"{val_metrics['token_acc']:.6f}",
				})
				eval_log_handle.flush()

			if _should_print_progress(step, config.steps, log_every, evaluated):
				postfix = {
					"it/s": f"{iter_per_sec:.1f}",
					"train_ce": f"{train_metrics['distill_ce']:.4f}",
				}
				if val_metrics is not None:
					postfix["val_ce"] = f"{val_metrics['distill_ce']:.4f}"
				progress.set_postfix(postfix)
				detail_fields = [
					f"tok/s={tokens_per_sec:.0f}",
					f"train_acc={train_metrics['teacher_acc']:.4f}",
				]
				if train_metrics["grad_norm"] is not None:
					detail_fields.append(f"grad={train_metrics['grad_norm']:.3f}")
				if param_norm is not None:
					detail_fields.append(f"param={param_norm:.3f}")
				if val_metrics is not None:
					detail_fields.append(f"val_acc={val_metrics['teacher_acc']:.4f}")
				progress_detail.set_description_str("metrics: " + " ".join(detail_fields))

			progress.update(1)

			steps_completed = step
	finally:
		progress.close()
		progress_detail.close()
		step_log_handle.close()
		eval_log_handle.close()
		batch_source.close()

	final_test = evaluate_student(student, teacher, test_tokens, config.temperature)
	summary = {
		"config": config.to_dict(),
		"status": status,
		"non_finite_step": non_finite_step,
		"steps_completed": steps_completed,
		"best_validation": best_validation,
		"final_validation": final_validation if final_validation is not None else evaluate_student(student, teacher, validation_tokens, config.temperature),
		"final_test": final_test,
		"elapsed_sec": round(perf_counter() - start_time, 6),
	}

	save_npz(run_dir / "student_weights.npz", student.copy_state_dict())
	_save_metrics_csv(run_dir / "metrics.csv", rows)
	save_json(run_dir / "train_summary.json", summary)
	_write_analysis_artifacts(run_dir, config, teacher, student, summary, rows)
	return summary


def main():
	parser = argparse.ArgumentParser(description="Train a student against a fixed teacher")
	parser.add_argument("--config", required=True, help="Path to a config JSON file")
	parser.add_argument("--run-name", default=None, help="Optional run name override")
	args = parser.parse_args()

	config = Config.load_json(args.config)
	if args.run_name is not None:
		config = config.clone(run_name=args.run_name)

	run_dir = Path("results") / f"run_{config.run_name}"
	summary = train_run(config, run_dir)
	print(f"Saved training artifacts to {run_dir}")
	print(f"validation_distill_ce={summary['final_validation']['distill_ce']:.6f}")
	print(f"test_distill_ce={summary['final_test']['distill_ce']:.6f}")


if __name__ == "__main__":
	main()
