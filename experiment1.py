import argparse

from experiment_utils import add_common_experiment_args, benchmark_run_dir, load_benchmark_from_args, print_experiment_summary, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the full AdamW vs Muon benchmark suite")
    add_common_experiment_args(parser)
    args = parser.parse_args()

    benchmark = load_benchmark_from_args(args, muon_mode="pure")
    run_dir = benchmark_run_dir(benchmark)
    summary = run_experiment(benchmark, run_dir, force=args.force, verify_existing=args.verify_existing)
    print_experiment_summary(run_dir, summary)


if __name__ == "__main__":
    main()
