import argparse
import csv
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter

import matplotlib
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_teacher_eval import build_teacher_and_eval
from eval import evaluate_saved_student
from muon_analysis.config import BenchmarkConfig, Config, TierConfig
from muon_analysis.io_utils import load_json, save_json
from train import train_run


def _count_total_trials(benchmark: BenchmarkConfig) -> int:
	total = 0
	for tier in benchmark.tiers:
		tuning = benchmark.tier_tuning(tier)
		total += len(benchmark.optimizers) * len(tuning.learning_rates) * len(tuning.weight_decays) * tier.repeats
	return total


def _slug_float(value: float) -> str:
	text = f"{value:.8g}"
	return text.replace("-", "m").replace(".", "p")


def _trial_dir(tier_dir: Path, optimizer: str, lr: float, weight_decay: float, repeat_index: int) -> Path:
	return tier_dir / "trials" / optimizer / f"lr_{_slug_float(lr)}__wd_{_slug_float(weight_decay)}" / f"repeat_{repeat_index + 1:02d}"


def _load_metrics_rows(path: Path) -> list[dict]:
	rows = []
	with path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			rows.append({key: float(value) for key, value in row.items()})
	return rows


def _mean(values: list[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
	if len(values) <= 1:
		return 0.0
	mean_value = _mean(values)
	variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
	return variance ** 0.5


def _max_grad_norm(trial_dir: Path) -> float:
	rows = _load_metrics_rows(trial_dir / "metrics.csv")
	return max(row["grad_norm"] for row in rows) if rows else 0.0


def _aggregate_curve(trial_dirs: list[Path]) -> list[dict]:
	per_trial_rows = [_load_metrics_rows(trial_dir / "metrics.csv") for trial_dir in trial_dirs]
	if not per_trial_rows:
		return []

	steps = [int(row["step"]) for row in per_trial_rows[0]]
	metrics = [key for key in per_trial_rows[0][0].keys() if key != "step"]
	aggregated = []
	for row_index, step in enumerate(steps):
		row = {"step": step}
		for metric in metrics:
			values = [trial_rows[row_index][metric] for trial_rows in per_trial_rows]
			row[f"mean_{metric}"] = _mean(values)
			row[f"std_{metric}"] = _std(values)
		aggregated.append(row)
	return aggregated


def _save_csv(path: Path, rows: list[dict]):
	if not rows:
		return
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def _plot_tier_curves(tier_dir: Path, optimizer_curves: dict[str, list[dict]]):
	plt.figure(figsize=(9, 5))
	for optimizer, rows in optimizer_curves.items():
		plt.plot([row["step"] for row in rows], [row["mean_val_distill_ce"] for row in rows], linewidth=2.0, label=optimizer)
	plt.xlabel("Step")
	plt.ylabel("Validation Distillation CE")
	plt.title("Validation Distillation CE vs Step")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(tier_dir / "plot_validation_vs_step.png", dpi=160)
	plt.close()

	plt.figure(figsize=(9, 5))
	for optimizer, rows in optimizer_curves.items():
		plt.plot([row["step"] for row in rows], [row["mean_train_distill_ce"] for row in rows], linewidth=2.0, label=optimizer)
	plt.xlabel("Step")
	plt.ylabel("Train Distillation CE")
	plt.title("Train Distillation CE vs Step")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(tier_dir / "plot_train_vs_step.png", dpi=160)
	plt.close()

	plt.figure(figsize=(9, 5))
	for optimizer, rows in optimizer_curves.items():
		plt.plot([row["mean_elapsed_sec"] for row in rows], [row["mean_val_distill_ce"] for row in rows], linewidth=2.0, label=optimizer)
	plt.xlabel("Elapsed Seconds")
	plt.ylabel("Validation Distillation CE")
	plt.title("Validation Distillation CE vs Time")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(tier_dir / "plot_validation_vs_time.png", dpi=160)
	plt.close()


def _plot_overall_test_loss(run_dir: Path, tier_summaries: list[dict], optimizers: list[str]):
	tier_names = [summary["tier"] for summary in tier_summaries]
	x_positions = list(range(len(tier_names)))
	bar_width = 0.8 / max(1, len(optimizers))

	plt.figure(figsize=(10, 5))
	for optimizer_index, optimizer in enumerate(optimizers):
		heights = [summary["best_configs"][optimizer]["final_test_distill_ce_mean"] for summary in tier_summaries]
		positions = [x + optimizer_index * bar_width for x in x_positions]
		plt.bar(positions, heights, width=bar_width, label=optimizer)

	center_positions = [x + bar_width * (len(optimizers) - 1) / 2 for x in x_positions]
	plt.xticks(center_positions, tier_names)
	plt.ylabel("Final Test Distillation CE")
	plt.title("Final Test Distillation CE by Tier and Optimizer")
	plt.grid(True, axis="y", alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(run_dir / "plot_final_test_loss_by_tier.png", dpi=160)
	plt.close()


def _write_tier_report(tier_dir: Path, tier_summary: dict):
	lines = [
		f"# Tier {tier_summary['tier']}",
		"",
		"## Best Hyperparameters",
		"",
	]
	for optimizer, summary in tier_summary["best_configs"].items():
		lines.append(
			f"- {optimizer}: lr={summary['lr']}, weight_decay={summary['weight_decay']}, "
			f"best validation distill CE={summary['best_validation_distill_ce_mean']:.6f} +/- {summary['best_validation_distill_ce_std']:.6f}, "
			f"final test distill CE={summary['final_test_distill_ce_mean']:.6f} +/- {summary['final_test_distill_ce_std']:.6f}, "
			f"diverged_runs={summary['diverged_runs']}"
		)
	lines.extend([
		"",
		"## Artifacts",
		"",
		"- plot_validation_vs_step.png",
		"- plot_train_vs_step.png",
		"- plot_validation_vs_time.png",
	])
	(tier_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_overall_report(run_dir: Path, benchmark: BenchmarkConfig, tier_summaries: list[dict]):
	lines = [
		f"# {benchmark.run_name}",
		"",
		"## Tier Summary",
		"",
	]
	for summary in tier_summaries:
		adamw = summary["best_configs"].get("adamw")
		muon = summary["best_configs"].get("muon")
		comparison = "n/a"
		if adamw is not None and muon is not None:
			gap = muon["final_test_distill_ce_mean"] - adamw["final_test_distill_ce_mean"]
			comparison = f"muon-adamw test gap={gap:.6f}"
		lines.append(f"- {summary['tier']}: {comparison}")
	lines.extend([
		"",
		"## Plot",
		"",
		"![Final Test Distillation CE by Tier](plot_final_test_loss_by_tier.png)",
	])
	(run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _single_build_config(benchmark_name: str, tier: TierConfig) -> Config:
	return tier.to_run_config(
		benchmark_name=benchmark_name,
		optimizer="adamw",
		lr=1e-3,
		weight_decay=0.0,
		repeat_index=0,
	)


def _read_last_logged_step(step_log_path: Path) -> int | None:
	if not step_log_path.exists():
		return None
	try:
		with step_log_path.open("rb") as handle:
			handle.seek(0, 2)
			size = handle.tell()
			if size == 0:
				return None
			handle.seek(-min(size, 4096), 2)
			chunk = handle.read().decode("utf-8", errors="ignore")
	except OSError:
		return None

	for line in reversed(chunk.splitlines()):
		if not line or not line[0].isdigit():
			continue
		parts = line.split(",", 1)
		if not parts:
			continue
		try:
			return int(parts[0])
		except ValueError:
			continue
	return None


class _BenchmarkProgressDisplay:
	def __init__(self, total_trials: int, worker_slots: int):
		self.main = tqdm(total=total_trials, desc="benchmark runs", dynamic_ncols=True, position=0)
		self.summary_line_1 = tqdm(total=0, desc="", bar_format="{desc}", dynamic_ncols=True, position=1)
		self.summary_line_2 = tqdm(total=0, desc="", bar_format="{desc}", dynamic_ncols=True, position=2)
		self.worker_bars = []
		for slot_index in range(worker_slots):
			bar = tqdm(total=1, desc=f"p{slot_index + 1}: idle", dynamic_ncols=True, position=3 + slot_index, leave=False)
			bar.n = 0
			bar.refresh()
			self.worker_bars.append(bar)

	def update_trial(self, tier: TierConfig, optimizer: str, lr: float, weight_decay: float):
		self.main.update(1)
		elapsed = float(self.main.format_dict.get("elapsed", 0.0))
		rate = self.main.format_dict.get("rate", None)
		rate_text = f"{rate:.2f} trials/s" if isinstance(rate, (int, float)) and rate and rate > 0 else "n/a"
		line_1 = (
			f"time: elapsed={elapsed:.1f}s done={int(self.main.n)}/{int(self.main.total or 0)} "
			f"speed={rate_text}"
		)
		line_2 = f"metrics: tier={tier.name} optimizer={optimizer} lr={lr:g} wd={weight_decay:g}"
		self.summary_line_1.set_description_str(line_1)
		self.summary_line_2.set_description_str(line_2)

	def start_worker_trial(self, slot_index: int, trial_label: str, total_steps: int):
		if slot_index >= len(self.worker_bars):
			return
		bar = self.worker_bars[slot_index]
		bar.reset(total=max(1, total_steps))
		bar.n = 0
		bar.set_description_str(f"p{slot_index + 1}: {trial_label}")
		bar.refresh()

	def refresh_worker_step(self, slot_index: int, step_log_path: Path):
		if slot_index >= len(self.worker_bars):
			return
		step = _read_last_logged_step(step_log_path)
		if step is None:
			return
		bar = self.worker_bars[slot_index]
		if step > bar.n:
			bar.n = min(step, int(bar.total))
			bar.refresh()

	def finish_worker_trial(self, slot_index: int, status_text: str):
		if slot_index >= len(self.worker_bars):
			return
		bar = self.worker_bars[slot_index]
		bar.n = int(bar.total)
		bar.set_description_str(f"p{slot_index + 1}: {status_text}")
		bar.refresh()

	def close(self):
		for bar in self.worker_bars:
			bar.close()
		self.summary_line_2.close()
		self.summary_line_1.close()
		self.main.close()


def _update_progress(progress_display: _BenchmarkProgressDisplay | None, tier: TierConfig, optimizer: str, lr: float, weight_decay: float):
	if progress_display is None:
		return
	progress_display.update_trial(tier, optimizer, lr, weight_decay)


def _run_trial(
	benchmark_name: str,
	tier: TierConfig,
	tier_dir: Path,
	optimizer: str,
	lr: float,
	weight_decay: float,
	repeat_index: int,
	force: bool,
	progress_bar=None,
	show_train_progress: bool = True,
) -> dict:
	trial_dir = _trial_dir(tier_dir, optimizer, lr, weight_decay, repeat_index)
	trial_dir.mkdir(parents=True, exist_ok=True)
	config = tier.to_run_config(
		benchmark_name=benchmark_name,
		optimizer=optimizer,
		lr=lr,
		weight_decay=weight_decay,
		repeat_index=repeat_index,
	)
	config.save_json(trial_dir / "config.json")

	summary_path = trial_dir / "train_summary.json"
	trial_label = f"{tier.name} {optimizer} lr={lr:g} wd={weight_decay:g} seed={repeat_index + 1}/{tier.repeats}"
	if force or not summary_path.exists():
		train_run(
			config,
			tier_dir,
			trial_dir,
			progress_desc=trial_label,
			leave_progress=False,
			show_progress=show_train_progress,
		)

	validation_metrics = evaluate_saved_student(config, tier_dir, trial_dir, split="validation")
	test_metrics = evaluate_saved_student(config, tier_dir, trial_dir, split="test")
	trial_summary = load_json(summary_path)
	trial_summary["final_validation"] = validation_metrics
	trial_summary["final_test"] = test_metrics
	trial_summary["max_grad_norm"] = _max_grad_norm(trial_dir)
	trial_summary["trial"] = {
		"optimizer": optimizer,
		"lr": lr,
		"weight_decay": weight_decay,
		"repeat_index": repeat_index,
		"trial_dir": str(trial_dir),
	}
	save_json(summary_path, trial_summary)
	_update_progress(progress_bar, tier, optimizer, lr, weight_decay)
	return trial_summary


def _run_trial_worker(payload: dict) -> dict:
	tier = TierConfig.from_dict(payload["tier"])
	return _run_trial(
		benchmark_name=payload["benchmark_name"],
		tier=tier,
		tier_dir=Path(payload["tier_dir"]),
		optimizer=payload["optimizer"],
		lr=payload["lr"],
		weight_decay=payload["weight_decay"],
		repeat_index=payload["repeat_index"],
		force=payload["force"],
		progress_bar=None,
		show_train_progress=False,
	)


def _run_trial_group(
	benchmark: BenchmarkConfig,
	tier: TierConfig,
	tier_dir: Path,
	optimizer: str,
	lr: float,
	weight_decay: float,
	force: bool,
	progress_bar=None,
	executor: ProcessPoolExecutor | None = None,
) -> list[dict]:
	if executor is None or min(benchmark.max_parallel_trials, tier.repeats) <= 1:
		return [
			_run_trial(
				benchmark_name=benchmark.run_name,
				tier=tier,
				tier_dir=tier_dir,
				optimizer=optimizer,
				lr=lr,
				weight_decay=weight_decay,
				repeat_index=repeat_index,
				force=force,
				progress_bar=progress_bar,
				show_train_progress=True,
			)
			for repeat_index in range(tier.repeats)
		]

	trials = [None] * tier.repeats
	max_workers = min(benchmark.max_parallel_trials, tier.repeats)
	worker_slots = min(max_workers, len(progress_bar.worker_bars)) if progress_bar is not None else 0
	next_repeat_index = 0
	running: dict = {}

	def _submit_one(slot_index: int) -> bool:
		nonlocal next_repeat_index
		if next_repeat_index >= tier.repeats:
			return False
		repeat_index = next_repeat_index
		next_repeat_index += 1
		payload = {
			"benchmark_name": benchmark.run_name,
			"tier": tier.to_dict(),
			"tier_dir": str(tier_dir),
			"optimizer": optimizer,
			"lr": lr,
			"weight_decay": weight_decay,
			"repeat_index": repeat_index,
			"force": force,
		}
		future = executor.submit(_run_trial_worker, payload)
		trial_dir = _trial_dir(tier_dir, optimizer, lr, weight_decay, repeat_index)
		running[future] = {
			"repeat_index": repeat_index,
			"slot_index": slot_index,
			"step_log": trial_dir / "step_log.csv",
		}
		if progress_bar is not None and slot_index < worker_slots:
			label = f"{optimizer} {tier.name} r{repeat_index + 1}/{tier.repeats}"
			progress_bar.start_worker_trial(slot_index, label, tier.steps)
		return True

	for slot_index in range(max_workers):
		if not _submit_one(slot_index):
			break

	while running:
		done, _ = wait(list(running.keys()), timeout=0.5, return_when=FIRST_COMPLETED)
		if progress_bar is not None:
			for state in running.values():
				slot_index = state["slot_index"]
				if slot_index < worker_slots:
					progress_bar.refresh_worker_step(slot_index, state["step_log"])
		if not done:
			continue

		for future in done:
			state = running.pop(future)
			repeat_index = state["repeat_index"]
			slot_index = state["slot_index"]
			trials[repeat_index] = future.result()
			if progress_bar is not None and slot_index < worker_slots:
				progress_bar.refresh_worker_step(slot_index, state["step_log"])
				progress_bar.finish_worker_trial(slot_index, f"done r{repeat_index + 1}/{tier.repeats}")
			_update_progress(progress_bar, tier, optimizer, lr, weight_decay)
			_submit_one(slot_index)

	return trials


def _summarize_config_trials(trials: list[dict]) -> dict:
	best_validation = [trial["best_validation"]["distill_ce"] for trial in trials]
	best_steps = [trial["best_validation"]["step"] for trial in trials]
	final_validation = [trial["final_validation"]["distill_ce"] for trial in trials]
	final_test = [trial["final_test"]["distill_ce"] for trial in trials]
	final_test_acc = [trial["final_test"]["teacher_acc"] for trial in trials]
	elapsed = [trial["elapsed_sec"] for trial in trials]
	max_grad = [trial["max_grad_norm"] for trial in trials]
	diverged_runs = sum(1 for trial in trials if trial["status"] != "ok")
	return {
		"lr": trials[0]["trial"]["lr"],
		"weight_decay": trials[0]["trial"]["weight_decay"],
		"best_validation_distill_ce_mean": _mean(best_validation),
		"best_validation_distill_ce_std": _std(best_validation),
		"best_step_mean": _mean(best_steps),
		"best_step_std": _std(best_steps),
		"final_validation_distill_ce_mean": _mean(final_validation),
		"final_validation_distill_ce_std": _std(final_validation),
		"final_test_distill_ce_mean": _mean(final_test),
		"final_test_distill_ce_std": _std(final_test),
		"final_test_teacher_acc_mean": _mean(final_test_acc),
		"final_test_teacher_acc_std": _std(final_test_acc),
		"elapsed_sec_mean": _mean(elapsed),
		"elapsed_sec_std": _std(elapsed),
		"max_grad_norm_mean": _mean(max_grad),
		"max_grad_norm_std": _std(max_grad),
		"diverged_runs": diverged_runs,
		"trial_dirs": [trial["trial"]["trial_dir"] for trial in trials],
	}


def _pick_best_config(config_summaries: list[dict]) -> dict:
	return min(
		config_summaries,
		key=lambda summary: (
			summary["best_validation_distill_ce_mean"],
			summary["best_validation_distill_ce_std"],
			summary["final_test_distill_ce_mean"],
		),
	)


def _run_tier(benchmark: BenchmarkConfig, tier: TierConfig, run_dir: Path, force: bool, verify_existing: bool, progress_bar: _BenchmarkProgressDisplay | None = None) -> dict:
	tier_dir = run_dir / tier.name
	tier_dir.mkdir(parents=True, exist_ok=True)
	build_config = _single_build_config(benchmark.run_name, tier)
	build_meta = build_teacher_and_eval(build_config, tier_dir, verify_existing=verify_existing and not force)
	tuning = benchmark.tier_tuning(tier)

	trial_groups: dict[str, list[dict]] = {}
	max_parallel = min(benchmark.max_parallel_trials, tier.repeats)
	parallel_executor = None
	if max_parallel > 1:
		parallel_executor = ProcessPoolExecutor(
			max_workers=max_parallel,
			mp_context=multiprocessing.get_context("spawn"),
		)
	try:
		for optimizer in benchmark.optimizers:
			for lr in tuning.learning_rates:
				for weight_decay in tuning.weight_decays:
					group_key = f"{optimizer}|{lr}|{weight_decay}"
					trial_groups[group_key] = _run_trial_group(
						benchmark,
						tier,
						tier_dir,
						optimizer,
						lr,
						weight_decay,
						force,
						progress_bar=progress_bar,
						executor=parallel_executor,
					)
	finally:
		if parallel_executor is not None:
			parallel_executor.shutdown(wait=True, cancel_futures=True)

	config_summaries_by_optimizer: dict[str, list[dict]] = {optimizer: [] for optimizer in benchmark.optimizers}
	for group_key, trials in trial_groups.items():
		optimizer = group_key.split("|", 1)[0]
		config_summaries_by_optimizer[optimizer].append(_summarize_config_trials(trials))

	best_configs = {}
	optimizer_curves = {}
	for optimizer, summaries in config_summaries_by_optimizer.items():
		best_summary = _pick_best_config(summaries)
		best_configs[optimizer] = best_summary
		trial_dirs = [Path(path) for path in best_summary["trial_dirs"]]
		curves = _aggregate_curve(trial_dirs)
		optimizer_curves[optimizer] = curves
		_save_csv(tier_dir / f"best_curve_{optimizer}.csv", curves)
		_save_csv(tier_dir / f"tuning_summary_{optimizer}.csv", summaries)

	_plot_tier_curves(tier_dir, optimizer_curves)
	tier_summary = {
		"tier": tier.name,
		"teacher_param_count": build_meta["derived"]["teacher_param_count"],
		"best_configs": best_configs,
	}
	save_json(tier_dir / "tier_summary.json", tier_summary)
	_write_tier_report(tier_dir, tier_summary)
	return tier_summary


def run_experiment(benchmark: BenchmarkConfig, run_dir: Path, force: bool = False, verify_existing: bool = False):
	run_dir.mkdir(parents=True, exist_ok=True)
	benchmark.save_json(run_dir / "benchmark_config_snapshot.json")
	start = perf_counter()
	tier_summaries = []
	total_trials = _count_total_trials(benchmark)
	worker_slots = max(0, min(benchmark.max_parallel_trials, 3))
	progress_bar = _BenchmarkProgressDisplay(total_trials=total_trials, worker_slots=worker_slots)
	try:
		for tier in benchmark.tiers:
			tier_summaries.append(_run_tier(benchmark, tier, run_dir, force=force, verify_existing=verify_existing, progress_bar=progress_bar))
	finally:
		progress_bar.close()

	_plot_overall_test_loss(run_dir, tier_summaries, benchmark.optimizers)
	_write_overall_report(run_dir, benchmark, tier_summaries)

	summary = {
		"benchmark": benchmark.to_dict(),
		"tiers": tier_summaries,
		"elapsed_sec": round(perf_counter() - start, 6),
	}
	save_json(run_dir / "experiment_summary.json", summary)
	return summary


def main():
	parser = argparse.ArgumentParser(description="Run the full AdamW vs Muon benchmark suite")
	parser.add_argument("--config", required=True, help="Path to a benchmark config JSON file")
	parser.add_argument("--run-name", default=None, help="Optional run name override")
	parser.add_argument("--max-parallel-trials", type=int, default=None, help="Optional override for the number of concurrent repeat trials")
	parser.add_argument("--verify-existing", action="store_true", help="Verify existing teacher artifacts before running")
	parser.add_argument("--force", action="store_true", help="Re-run trials even when outputs already exist")
	args = parser.parse_args()

	benchmark = BenchmarkConfig.load_json(args.config)
	if args.run_name is not None:
		benchmark.run_name = args.run_name
	if args.max_parallel_trials is not None:
		benchmark.max_parallel_trials = args.max_parallel_trials

	run_dir = Path(benchmark.output_root) / f"run_{benchmark.run_name}"
	summary = run_experiment(benchmark, run_dir, force=args.force, verify_existing=args.verify_existing)

	print(f"Saved full experiment artifacts to {run_dir}")
	for tier_summary in summary["tiers"]:
		for optimizer, config_summary in tier_summary["best_configs"].items():
			print(
				f"{tier_summary['tier']} {optimizer}: "
				f"val={config_summary['best_validation_distill_ce_mean']:.6f} "
				f"test={config_summary['final_test_distill_ce_mean']:.6f}"
			)


if __name__ == "__main__":
	main()