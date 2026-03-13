import argparse
import os
from pathlib import Path
import tempfile

import cupy as cp
import numpy as np

from muon_analysis.config import Config
from muon_analysis.dtypes import resolve_dtype
from muon_analysis.io_utils import load_npz, load_tokens, save_json, save_npz, save_tokens
from muon_analysis.models.model_utils import ctx_to_onehot_concat
from muon_analysis.models.teacher import Teacher


def _teacher_signal_stats(teacher: Teacher, tokens: cp.ndarray, config: Config, dtype, max_windows: int = 65536) -> dict:
    from muon_analysis.models.model_utils import make_windows, stable_softmax

    ctx, _ = make_windows(tokens, config.K)
    if ctx.shape[0] > max_windows:
        ctx = ctx[:max_windows]

    u = ctx_to_onehot_concat(ctx, config.V, dtype)
    logits = teacher.forward_logits(u).astype(cp.float32, copy=False)
    probs = stable_softmax(logits / float(config.temperature), axis=1)

    max_prob = cp.max(probs, axis=1)
    entropy = -cp.sum(probs * cp.log(cp.clip(probs, 1e-12, 1.0)), axis=1)
    logit_rms = cp.sqrt(cp.mean(logits * logits, axis=1))
    ln_vocab = cp.log(cp.asarray(config.V, dtype=cp.float32))

    mean_entropy = float(cp.mean(entropy).get())
    mean_max_prob = float(cp.mean(max_prob).get())
    mean_logit_rms = float(cp.mean(logit_rms).get())
    uniform_entropy = float(ln_vocab.get())
    entropy_gap = uniform_entropy - mean_entropy
    max_prob_ratio_to_uniform = mean_max_prob / (1.0 / float(config.V))

    return {
        "window_count": int(ctx.shape[0]),
        "mean_entropy": mean_entropy,
        "uniform_entropy": uniform_entropy,
        "entropy_gap": float(entropy_gap),
        "mean_max_prob": mean_max_prob,
        "max_prob_ratio_to_uniform": float(max_prob_ratio_to_uniform),
        "mean_logit_rms": mean_logit_rms,
    }


def _assert_teacher_signal_quality(stats: dict, config: Config):
    min_entropy_gap = 1e-4
    min_max_prob_ratio = 1.04
    min_logit_rms = 1e-3

    if (
        stats["entropy_gap"] < min_entropy_gap
        or stats["max_prob_ratio_to_uniform"] < min_max_prob_ratio
        or stats["mean_logit_rms"] < min_logit_rms
    ):
        raise ValueError(
            "Teacher soft targets are too close to uniform; distillation signal is likely too weak for optimizer benchmarking. "
            f"tier={config.tier_name}, V={config.V}, temperature={config.temperature}, "
            f"entropy_gap={stats['entropy_gap']:.6g} (min {min_entropy_gap}), "
            f"max_prob_ratio={stats['max_prob_ratio_to_uniform']:.6g} (min {min_max_prob_ratio}), "
            f"mean_logit_rms={stats['mean_logit_rms']:.6g} (min {min_logit_rms}). "
            "Try lower temperature, different teacher seed, or stronger teacher initialization scale."
        )


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


def train_tokens_path(artifact_dir: Path, train_seed: int) -> Path:
    return artifact_dir / "train_tokens" / f"seed_{int(train_seed)}.npy"


def _validate_existing_train_tokens_shape(path: Path, config: Config):
    cached = np.load(path, mmap_mode="r")
    expected_shape = (int(config.steps), int(config.B_train), int(config.T))
    assert cached.shape == expected_shape, f"Train tokens shape mismatch at {path}: {cached.shape} != {expected_shape}"
    assert cached.dtype == np.int32, f"Train tokens dtype mismatch at {path}: {cached.dtype} != int32"


def _verify_existing_train_tokens_head(teacher: Teacher, config: Config, path: Path, train_seed: int):
    cached = np.load(path, mmap_mode="r")
    if cached.shape[0] == 0:
        return
    expected_step_one = teacher.generate_sequences(
        config.B_train,
        config.T,
        seed=int(train_seed) + 1,
        prefix_mode=config.prefix_mode,
        bos_id=config.bos_id,
        temperature=config.temperature,
    )
    cp.cuda.get_current_stream().synchronize()
    assert np.array_equal(cached[0], cp.asnumpy(expected_step_one)), f"Train tokens first step mismatch at {path}"


def _write_train_tokens_file(teacher: Teacher, config: Config, path: Path, train_seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(fd)

    try:
        mmap = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.int32,
            shape=(int(config.steps), int(config.B_train), int(config.T)),
        )
        block_size = max(1, min(int(config.train_batch_block_size), int(config.steps)))
        for start_step in range(1, int(config.steps) + 1, block_size):
            block_steps = min(block_size, int(config.steps) - start_step + 1)
            total_batch = int(config.B_train * block_steps)
            tokens = teacher.generate_sequences(
                total_batch,
                config.T,
                seed=int(train_seed) + start_step,
                prefix_mode=config.prefix_mode,
                bos_id=config.bos_id,
                temperature=config.temperature,
            )
            cp.cuda.get_current_stream().synchronize()
            mmap[start_step - 1:start_step - 1 + block_steps] = cp.asnumpy(tokens.reshape(block_steps, config.B_train, config.T))

        mmap.flush()
        del mmap
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_train_tokens(
    config: Config,
    artifact_dir: Path,
    train_seeds: list[int],
    force: bool = False,
    verify_existing: bool = False,
    progress_callback=None,
):
    teacher_state = load_npz(artifact_dir / "teacher_weights.npz")
    teacher = Teacher(
        config.V,
        config.K,
        config.width,
        config.layers,
        activation=config.activation,
        dtype=resolve_dtype(config.dtype),
        params=teacher_state,
    )

    manifest = []
    total = len(train_seeds)
    for seed_index, train_seed in enumerate(train_seeds, start=1):
        path = train_tokens_path(artifact_dir, train_seed)
        if progress_callback is not None:
            progress_callback(seed_index, total, int(train_seed), "checking")
        if path.exists() and not force:
            _validate_existing_train_tokens_shape(path, config)
            if verify_existing:
                _verify_existing_train_tokens_head(teacher, config, path, train_seed)
            if progress_callback is not None:
                progress_callback(seed_index, total, int(train_seed), "existing")
            manifest.append({"train_seed": int(train_seed), "path": str(path), "status": "existing"})
            continue

        if progress_callback is not None:
            progress_callback(seed_index, total, int(train_seed), "writing")
        _write_train_tokens_file(teacher, config, path, train_seed)
        if progress_callback is not None:
            progress_callback(seed_index, total, int(train_seed), "written")
        manifest.append({"train_seed": int(train_seed), "path": str(path), "status": "written"})

    save_json(artifact_dir / "train_tokens_manifest.json", {"entries": manifest})
    return manifest


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
    teacher_signal = _teacher_signal_stats(teacher, validation_tokens, config, dtype)
    _assert_teacher_signal_quality(teacher_signal, config)

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
        "teacher_signal": teacher_signal,
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