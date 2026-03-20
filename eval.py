import argparse
from pathlib import Path

from muon_analysis.config import Config
from muon_analysis.dtypes import resolve_dtype
from muon_analysis.io_utils import load_npz, load_tokens, save_json
from muon_analysis.models.student import Student
from muon_analysis.models.teacher import Teacher
from train import evaluate_student


def evaluate_saved_student(config: Config, artifact_dir: Path, run_dir: Path | None = None, split: str = "test") -> dict:
	run_dir = artifact_dir if run_dir is None else run_dir
	dtype = resolve_dtype(config.dtype)
	teacher_state = load_npz(artifact_dir / "teacher_weights.npz")
	student_state = load_npz(run_dir / "student_weights.npz")

	if split == "validation":
		tokens = load_tokens(artifact_dir / "validation_tokens.npy")
	elif split == "test":
		tokens = load_tokens(artifact_dir / "test_tokens.npy")
	else:
		raise ValueError(f"Unsupported split {split}")

	# Rebuild both models from saved weights for deterministic evaluation.
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
		params=student_state,
	)

	# Reuse the same evaluator as training for metric consistency.
	metrics = evaluate_student(student, teacher, tokens, config.temperature)
	save_json(run_dir / f"eval_{split}.json", metrics)
	return metrics


def main():
	parser = argparse.ArgumentParser(description="Evaluate a saved student on the fixed validation or test set")
	parser.add_argument("--config", required=True, help="Path to a config JSON file")
	parser.add_argument("--split", default="test", choices=["validation", "test"], help="Dataset split to evaluate")
	parser.add_argument("--run-name", default=None, help="Optional run name override")
	args = parser.parse_args()

	config = Config.load_json(args.config)
	if args.run_name is not None:
		config = config.clone(run_name=args.run_name)

	run_dir = Path("results") / f"run_{config.run_name}"
	metrics = evaluate_saved_student(config, run_dir, split=args.split)
	print(f"{args.split}_distill_ce={metrics['distill_ce']:.6f}")
	print(f"{args.split}_teacher_acc={metrics['teacher_acc']:.6f}")


if __name__ == "__main__":
	main()
