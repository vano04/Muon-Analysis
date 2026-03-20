# Muon-Analysis

CuPy-only teacher-student autoregressive MLP experiments for comparing AdamW and Muon without PyTorch.

Install required packages with:
```bash
pip install -r requirements.txt
```

## Layout

- `muon_analysis/`: reusable package code
- `muon_analysis/models/`: bias-free autoregressive MLP, teacher, student, and model utilities
- `configs/`: checked-in benchmark suite configs for smoke, small, medium, large, and the full suite
- `build_teacher_eval.py`: build a fixed teacher plus validation/test sets for a single tier
- `train.py`: run one optimizer/lr/wd/seed trial against shared tier artifacts
- `eval.py`: evaluate a saved trial on the fixed validation or test split
- `experiment1.py`: run the full multi-tier AdamW-vs-Muon benchmark with tuning and repeated seeds
- `experiment2.py`: run the AdamW-vs-Muon-hybrid benchmark variant
- `results/`: generated teachers, trial logs, plots, summaries, and reports

## Example

Run the full suite:

```bash
python experiment1.py --config configs/benchmark_suite.json
```

Run one tier only:

```bash
python experiment1.py --config configs/benchmark_medium.json
```

Run a smoke benchmark:

```bash
python experiment1.py --config configs/smoke.json
```

Single-trial utilities remain available when you want to inspect one exact setting:

```bash
python build_teacher_eval.py --config configs/benchmark_small.json --run-name debug_trial
python train.py --config configs/benchmark_500m_bf16_adamw.json --run-name legacy_single_trial
python eval.py --config configs/benchmark_500m_bf16_adamw.json --run-name legacy_single_trial --split test
```

Multi process execution:
```bash
python experiment1.py --config benchmark_suite.json --max-parallel-trials 3 --force
```

`experiment.py` remains as a compatibility shim that forwards to `experiment1.py`.

## Benchmark Structure

Each benchmark config defines:

- one or more model size tiers
- a fair learning-rate and weight-decay sweep shared across AdamW and Muon
- repeated student initialization and training-stream seeds
- fixed teacher, validation, and test seeds per tier

The experiment runner selects hyperparameters using validation distillation loss only, then reports the repeated-seed metrics for the selected setting.

## Threading

Tier configs accept `num_threads` to enable a small amount of safe multithreading. When `num_threads > 1`, training uses a background thread to prefetch the next teacher batch and uses extra threads for plot generation.

## Train Stream Reuse

During `experiment.py` runs, each tier now pre-generates train token streams once per repeat seed and saves them under `train_tokens/seed_<train_seed>.npy` inside the tier folder. All optimizer/lr/wd trials for that repeat reuse the same saved stream, removing per-trial teacher sampling overhead while preserving the same deterministic training data.

## Dtypes

Configs accept canonical dtype names `float16`, `float32`, `float64` and aliases such as `fp16`, `fp32`, `fp64`, and `bf16`.

`bfloat16`/`bf16` is only accepted when the active CuPy runtime actually exposes that dtype.
