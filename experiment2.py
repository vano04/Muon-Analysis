import argparse
from pathlib import Path

from muon_analysis.config import BenchmarkConfig

from experiment import run_experiment


def _to_hybrid_optimizer(name: str) -> str:
	return "muon_hybrid" if str(name).lower() == "muon" else name


def main():
	parser = argparse.ArgumentParser(description="Run the benchmark suite with Muon configured as a hybrid optimizer")
	parser.add_argument("--config", required=True, help="Path to a benchmark config JSON file")
	parser.add_argument("--run-name", default=None, help="Optional run name override")
	parser.add_argument("--max-parallel-trials", type=int, default=None, help="Optional override for the number of concurrent repeat trials")
	parser.add_argument("--verify-existing", action="store_true", help="Verify existing teacher artifacts before running")
	parser.add_argument("--force", action="store_true", help="Re-run trials even when outputs already exist")
	args = parser.parse_args()

	benchmark = BenchmarkConfig.load_json(args.config)
	if args.run_name is not None:
		benchmark.run_name = args.run_name
	else:
		benchmark.run_name = f"{benchmark.run_name}_hybrid"
	if args.max_parallel_trials is not None:
		benchmark.max_parallel_trials = args.max_parallel_trials

	benchmark.optimizers = [_to_hybrid_optimizer(name) for name in benchmark.optimizers]
	benchmark.validate()

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
