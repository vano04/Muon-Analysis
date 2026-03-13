import argparse
from pathlib import Path

import cupy as cp

from muon_analysis.config import Config
from muon_analysis.dtypes import resolve_dtype
from muon_analysis.io_utils import load_npz, load_tokens, save_json, save_npz, save_tokens
from muon_analysis.models.model_utils import ctx_to_onehot_concat
from muon_analysis.models.teacher import Teacher


def _param_shapes(state_dict: dict) -> dict:
    return {key: list(value.shape) for key, value in state_dict.items()}


def _build_split_tokens(teacher: Teacher, config: Config, split: str) -> cp.ndarray:
    if split == "validation":
        batch_size = config.B_val
        seed = config.validation_seed
    elif split == "test":
        batch_size = config.B_test
        seed = config.test_seed
    else:
        raise ValueError(f"Unsupported split {split}")

    return teacher.generate_sequences(
        batch_size,
        config.T,
        seed=seed,
        prefix_mode=config.prefix_mode,
        bos_id=config.bos_id,
        temperature=config.temperature,
    )


def build_teacher_and_eval(config: Config, artifact_dir: Path, verify_existing: bool = False):
    dtype = resolve_dtype(config.dtype)
    teacher = Teacher(
        config.V,
        config.K,
        config.width,
        config.layers,
        activation=config.activation,
        dtype=dtype,
        seed=config.teacher_seed,
    )

    validation_tokens = _build_split_tokens(teacher, config, "validation")
    test_tokens = _build_split_tokens(teacher, config, "test")

    sample_ctx = validation_tokens[: min(config.B_val, 4), : config.K]
    sample_u = ctx_to_onehot_concat(sample_ctx, config.V, dtype)
    sample_logits = teacher.forward_logits(sample_u)

    assert cp.isfinite(sample_logits).all(), "Teacher logits contain non-finite values"
    validation_hist = cp.bincount(validation_tokens.reshape(-1), minlength=config.V)
    test_hist = cp.bincount(test_tokens.reshape(-1), minlength=config.V)
    assert int((validation_hist > 0).sum().get()) > 1, "Validation set token histogram is degenerate"
    assert int((test_hist > 0).sum().get()) > 1, "Test set token histogram is degenerate"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = artifact_dir / "teacher_weights.npz"
    validation_path = artifact_dir / "validation_tokens.npy"
    test_path = artifact_dir / "test_tokens.npy"
    meta_path = artifact_dir / "meta.json"

    state_dict = teacher.copy_state_dict()
    derived = {
        "in_dim": config.in_dim,
        "teacher_param_count": teacher.parameter_count(),
        "teacher_param_shapes": _param_shapes(state_dict),
        "validation_token_histogram": cp.asnumpy(validation_hist).tolist(),
        "test_token_histogram": cp.asnumpy(test_hist).tolist(),
    }

    meta = {
        "config": config.to_dict(),
        "derived": derived,
    }

    if verify_existing and teacher_path.exists() and validation_path.exists() and test_path.exists():
        prev_teacher = load_npz(teacher_path)
        prev_validation = load_tokens(validation_path)
        prev_test = load_tokens(test_path)

        for key, value in state_dict.items():
            assert cp.array_equal(prev_teacher[key], value), f"Teacher mismatch for {key}"
        assert cp.array_equal(prev_validation, validation_tokens), "Validation tokens mismatch"
        assert cp.array_equal(prev_test, test_tokens), "Test tokens mismatch"

    save_npz(teacher_path, state_dict)
    save_tokens(validation_path, validation_tokens)
    save_tokens(test_path, test_tokens)
    save_json(meta_path, meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build a fixed teacher plus validation and test sets")
    parser.add_argument("--config", required=True, help="Path to a config JSON file")
    parser.add_argument("--run-name", default=None, help="Optional run name override")
    parser.add_argument("--verify-existing", action="store_true", help="Verify existing artifacts match")
    args = parser.parse_args()

    config = Config.load_json(args.config)
    if args.run_name is not None:
        config = config.clone(run_name=args.run_name)

    run_dir = Path("results") / f"run_{config.run_name}"
    meta = build_teacher_and_eval(config, run_dir, verify_existing=args.verify_existing)
    print(f"Saved teacher and eval sets to {run_dir}")
    print(f"Teacher params: {meta['derived']['teacher_param_count']}")


if __name__ == "__main__":
    main()