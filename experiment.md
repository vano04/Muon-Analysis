# AdamW vs Muon Experiment Design
## Agent-Facing Implementation Spec

This document specifies how to design and implement a **controlled optimizer benchmark** comparing **AdamW** and **Muon** on a **teacher-student autoregressive bias-free MLP** using **soft targets**. The goal is to isolate optimizer behavior as cleanly as possible while keeping the setup simple, reproducible, and scalable.

---

# 1. Objective

Implement an experiment that answers:

> On a synthetic teacher-student autoregressive next-token prediction task with soft targets, how does **Muon** compare to **AdamW** in terms of:
- convergence speed,
- final performance,
- training stability,
- and scaling behavior as model size increases?

This experiment is **not** about hybrid optimizers. It is strictly:

- **AdamW-only**
- **Muon-only**

Each run must use exactly one optimizer for all trainable parameters included in the benchmark.

---

# 2. High-Level Design

Use a **synthetic autoregressive teacher-student setup**:

- A fixed **teacher MLP_AR** generates next-token logits for synthetic token sequences.
- A trainable **student MLP_AR** learns from the teacher using **soft targets**.
- The student is trained with either **AdamW** or **Muon**.
- Repeat this benchmark for **small**, **medium**, and **large** models.

The optimizer should be the main independent variable.

Everything else should be held constant or controlled.

---

# 3. Core Principles

## 3.1 Controlled Comparison
The following should be identical across AdamW and Muon runs unless explicitly part of a tuning sweep:

- teacher model
- dataset generation process
- train/validation/test splits
- architecture for a given size tier
- batch size
- sequence length
- vocabulary size
- initialization scheme
- number of training steps
- evaluation cadence
- seed protocol
- hardware/device type

## 3.2 Soft Targets
Use the **teacher’s full probability distribution** as the training target rather than hard labels.

Preferred loss:
- **KL divergence**
or
- **cross-entropy with teacher softmax probabilities**

Soft targets are preferred because they:
- reduce label noise,
- give smoother gradients,
- make optimizer differences easier to observe,
- and better reflect teacher-student distillation.

## 3.3 No Hybrid Logic
Do **not** mix update rules inside a single run.
Examples of what not to do:
- Muon for matrices and AdamW for embeddings in the same experiment run
- AdamW fallback for unsupported tensors unless this is a separate explicitly documented variant

For this benchmark, keep the architecture simple enough that both optimizers can be applied fairly.

---

# 4. Task Definition

## 4.1 Synthetic Autoregressive Task
Construct token sequences from a synthetic discrete vocabulary.

At each timestep:
- the student receives a prefix of tokens,
- predicts logits for the next token,
- and is trained to match the teacher’s output distribution.

## 4.2 Suggested Input/Output Setup
Let:
- `V` = vocabulary size
- `T` = sequence length / context length
- `B` = batch size

A sequence is:
`x = [x_0, x_1, ..., x_{T-1}]`

For each position `t`, the model predicts the distribution of `x_t` from the prefix:
`[x_0, ..., x_{t-1}]`

A standard training batch may produce:
- input tokens of shape `(B, T)`
- next-token logits of shape `(B, T, V)`

Use causal / autoregressive behavior only.

---

# 5. Model Architecture

Use the same architecture family for teacher and student.

## 5.1 Required Model Properties
- autoregressive
- MLP-based, not attention-based
- bias-free linear layers
- token embedding layer
- hidden MLP stack
- output projection to vocabulary logits

Keep the architecture simple and consistent.

## 5.2 Bias-Free Requirement
Prefer **bias-free** linear layers throughout if possible.
This keeps the setup cleaner for Muon and reduces optimizer-specific incompatibility.

## 5.3 Teacher vs Student
Use the same overall architecture family for both.

Recommended:
- same depth and form,
- but teacher may optionally be wider than student if desired.

However, for a pure optimizer benchmark, the cleanest setup is:

- fixed teacher architecture per size tier
- fixed student architecture per size tier
- same teacher for all runs in that tier

The teacher must be **frozen**.

---

# 6. Model Size Tiers

Implement three benchmark scales.

These do not need to be exactly these values, but they should be in this spirit.

## 6.1 Small
Purpose:
- sanity check
- debugging
- verify all codepaths work
- initial optimizer behavior

Suggested:
- width: `64`
- layers: `3`
- vocab size: `16` or `32`
- context length: `32`

## 6.2 Medium
Purpose:
- main benchmark
- likely first tier where optimizer behavior becomes meaningfully different

Suggested:
- width: `128`
- layers: `4`
- vocab size: `32`
- context length: `32` or `48`

## 6.3 Large
Purpose:
- test scaling claim
- determine whether Muon becomes more favorable at larger width / parameter count

Suggested:
- width: `256`
- layers: `6`
- vocab size: `32` or `64`
- context length: `48` or `64`

## 6.4 Important Note
Do not change too many variables at once.
The main scale variable should be **model size**.

Keep vocabulary and context either:
- fixed across all tiers, or
- only lightly adjusted if memory/runtime requires it.

---

# 7. Teacher Construction

## 7.1 Fixed Teacher Per Tier
For each size tier:
1. initialize one teacher with a fixed seed,
2. freeze it,
3. use it to generate soft targets for all optimizer runs in that tier.

This ensures the task is identical for AdamW and Muon.

## 7.2 Teacher Temperature
Use a fixed softmax temperature when converting teacher logits to target probabilities.

Recommended starting point:
- `temperature = 1.0`

Possible optional ablation later:
- `temperature = 0.7`
- `temperature = 1.0`
- `temperature = 1.3`

But for the initial benchmark, keep temperature fixed.

## 7.3 Teacher Output Storage
Two options:

### Option A: Online Teacher Forward
At each batch:
- generate synthetic input sequences,
- run teacher forward pass,
- compute soft targets on the fly.

Pros:
- simple
- low storage cost

Cons:
- more compute each step

### Option B: Precomputed Teacher Targets
Pre-generate the train/val/test datasets with teacher logits or probabilities.

Pros:
- faster student training runs
- perfectly deterministic

Cons:
- more storage
- more preprocessing logic

Recommendation:
- start with **online teacher forward** for simplicity,
- optionally move to precomputed targets later if needed.

---

# 8. Dataset Generation

## 8.1 Synthetic Token Sequences
Generate sequences from a simple discrete token process.

Recommended starting approach:
- uniformly sample tokens from `[0, V-1]`
- include BOS token if desired
- keep generation process fixed across all runs

The key is not to make the dataset realistic.
The key is to make it controlled and reproducible.

## 8.2 Splits
Create separate:
- training set
- validation set
- test set

Possible approaches:
- fixed number of synthetic sequences per split
- or infinite synthetic stream with fixed validation/test sets

Recommended:
- training stream may be generated from a seed-driven RNG
- validation and test sets should be fixed and reusable

## 8.3 Suggested Split Sizes
Adjust to runtime constraints, but something like:

- train: large stream or many batches
- validation: enough batches for stable averages
- test: enough batches for stable final estimates

A simple starting point:
- validation: 100-200 batches
- test: 100-200 batches

---

# 9. Loss Function

## 9.1 Preferred Loss
Train student to match teacher soft targets.

Let:
- `z_teacher` = teacher logits
- `z_student` = student logits
- `p_teacher = softmax(z_teacher / temp)`
- `p_student = softmax(z_student / temp)`

Use one of:

### Preferred
**KL divergence**
`KL(p_teacher || p_student)`

### Acceptable
Cross-entropy between teacher probabilities and student log probabilities.

## 9.2 Temperature Handling
If using temperature in both teacher and student distillation loss, keep it fixed across all runs.
Do not introduce optimizer-specific temperature settings.

---

# 10. Independent Variable

The only primary independent variable is:

- optimizer = `adamw` or `muon`

Everything else should be kept fixed within a given comparison block except for explicitly defined tuning sweeps.

---

# 11. Recommended Hyperparameter Strategy

Do **not** compare AdamW and Muon with one arbitrary shared learning rate.

Each optimizer should receive a modest but fair tuning sweep.

## 11.1 Minimum Tuning Sweep
For each optimizer and each size tier, tune:
- learning rate
- weight decay

Optional:
- optimizer-specific momentum / beta values only if needed

## 11.2 Suggested Learning Rate Search
Use a compact grid or log sweep.

Example:
- `1e-4`
- `3e-4`
- `1e-3`
- `3e-3`
- `1e-2`

For larger models, you may narrow based on early results.

## 11.3 Suggested Weight Decay Search
Example:
- `0.0`
- `1e-4`
- `5e-4`
- `1e-3`

## 11.4 Tuning Rule
Use validation loss to select the best hyperparameters.
Then report final test results using the best validation configuration.

Do not pick hyperparameters based on test performance.

---

# 12. Seed Protocol

Use a structured seed policy.

## 12.1 Recommended Seeds
Separate seeds for:
- teacher initialization
- student initialization
- training data stream
- evaluation data
- any stochastic sampling

## 12.2 Fixed vs Varying
Recommended policy:

### Fixed per size tier
- teacher seed
- validation set seed
- test set seed

### Repeated across runs
- student init seed
- training stream seed

## 12.3 Repeats
Run multiple seeds per optimizer configuration.

Recommended:
- minimum: `3` seeds
- better: `5` seeds if runtime permits

Report:
- mean
- standard deviation

This is important because optimizer conclusions from one seed are weak.

---

# 13. Training Budget

Use a fixed training budget per size tier.

## 13.1 Preferred Comparison Budget
Use:
- fixed number of training steps

This is easier and cleaner than fixed wall-clock time.

## 13.2 Example Training Steps
These are placeholders and may be adjusted based on runtime:

- small: `5,000` to `10,000` steps
- medium: `10,000` to `20,000` steps
- large: `20,000` to `40,000` steps

The exact number is less important than:
- being the same across optimizer runs for that comparison,
- and long enough to show meaningful convergence trends.

## 13.3 Evaluation Cadence
Evaluate every fixed number of steps.

Example:
- every `100` or `200` steps

Always use the same eval cadence within a benchmark block.

---

# 14. Metrics to Record

## 14.1 Primary Metrics
For each run, log:
- training loss vs step
- validation loss vs step
- final validation loss
- final test loss

## 14.2 Convergence Metrics
Also compute:
- best validation loss achieved
- step at which a target validation loss is reached
- step at which best validation loss occurs

If feasible, define a threshold loss and measure:
- steps to threshold
- wall-clock time to threshold

## 14.3 Stability Metrics
Record:
- NaN occurrence
- divergence
- gradient norm spikes
- parameter norm growth
- failed runs

## 14.4 Efficiency Metrics
If available, also log:
- examples/sec
- tokens/sec
- wall-clock training time

This helps distinguish:
- better optimization trajectory
vs
- simply more expensive updates

---

# 15. Outputs and Artifacts

The agent should save all raw and summarized outputs needed for analysis.

## 15.1 Per-Run Outputs
For each run, save:
- config file
- optimizer name
- seed values
- train/val/test metrics over time
- final summary metrics

Suggested format:
- JSON
- CSV
- or NPZ

## 15.2 Aggregated Outputs
For each size tier, produce:
- averaged learning curves across seeds
- summary table for AdamW vs Muon
- best hyperparameter table

## 15.3 Plots
At minimum generate:

### Plot A
Validation loss vs training step

### Plot B
Training loss vs training step

### Plot C
Validation loss vs wall-clock time

### Plot D
Final test loss by model size and optimizer

Optional:
- gradient norm vs step
- parameter norm vs step

---

# 16. Main Comparison Structure

The cleanest structure is:

For each size tier:
1. define teacher
2. define student architecture
3. tune AdamW
4. tune Muon
5. run best config over multiple seeds
6. aggregate results
7. compare convergence, final loss, and stability

Then compare across:
- small
- medium
- large

This directly answers whether optimizer differences become more or less pronounced with scale.

---

# 17. Recommended Experiment Phases

## Phase 1: Implementation Sanity Check
Goal:
- verify forward pass
- verify loss works
- verify both optimizers can train without error
- verify metrics/logging/plots function

Use:
- small model only
- 1 seed
- short training run

This phase is not for final claims.

## Phase 2: Small Benchmark
Goal:
- initial AdamW vs Muon comparison
- verify trends are consistent

Use:
- small model
- modest tuning sweep
- 3 seeds on best config

## Phase 3: Medium Benchmark
Goal:
- main comparison
- likely strongest evidence for report

Use:
- medium model
- same protocol

## Phase 4: Large Benchmark
Goal:
- test scaling hypothesis
- see if Muon improves relative to AdamW as size increases

Use:
- large model
- same protocol
- may reduce tuning grid if needed, but keep it fair

---

# 18. Fairness Requirements

The agent must enforce the following:

## 18.1 Same Architecture Per Comparison
Within each size tier, AdamW and Muon must train the exact same student architecture.

## 18.2 Same Teacher
Use the exact same teacher weights for both optimizers within a tier.

## 18.3 Same Data
Use the same validation and test sets.
Prefer matching training stream seeds too unless varying them across repeated trials.

## 18.4 Same Budget
Same number of steps and eval intervals.

## 18.5 Comparable Tuning Effort
Do not heavily tune one optimizer and barely tune the other.

---

# 19. What Not to Do

The agent should avoid the following mistakes:

- comparing one AdamW LR against one Muon LR and calling it definitive
- changing architecture between optimizers
- using hard labels instead of soft targets for the main experiment
- using only one random seed
- reporting only training loss and not validation/test results
- drawing conclusions from the small model only
- silently falling back to a different optimizer for unsupported tensors
- mixing hybrid optimizer logic in the main benchmark

---

# 20. Recommended Default Configuration

This is a starting recommendation, not a hard requirement.

## Shared
- soft targets: yes
- fixed teacher per size tier: yes
- eval every: `100` steps
- repeats: `3` seeds minimum
- optimizers: `adamw`, `muon`

## Small
- width: `64`
- layers: `3`
- vocab: `16` or `32`
- context: `32`
- batch size: `96`
- steps: `5,000` to `10,000`

## Medium
- width: `128`
- layers: `4`
- vocab: `32`
- context: `32` or `48`
- batch size: `96` or `128`
- steps: `10,000` to `20,000`

## Large
- width: `256`
- layers: `6`
- vocab: `32` or `64`
- context: `48` or `64`
- batch size: lower if memory requires
- steps: `20,000` to `40,000`

## Tuning Grid
Learning rate:
- `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`

Weight decay:
- `0.0`, `1e-4`, `5e-4`, `1e-3`

These values may be adjusted after Phase 1.

---

# 21. Deliverables Expected from the Agent

The agent should produce:

## 21.1 Implementation
- experiment runner
- config system
- teacher/student model definitions if not already present
- AdamW and Muon training paths
- evaluation pipeline
- logging and checkpointing

## 21.2 Result Artifacts
- per-run logs
- aggregate metrics
- plots
- tables for report inclusion

## 21.3 Final Summary
A concise written summary that answers:
- which optimizer converged faster?
- which optimizer achieved lower final validation/test loss?
- how stable was each optimizer?
- did the optimizer gap change from small to medium to large?

---

# 22. Suggested Directory Structure

```text
project/
  configs/
    small_adamw.json
    small_muon.json
    medium_adamw.json
    medium_muon.json
    large_adamw.json
    large_muon.json

  models/
    mlp_ar.py
    teacher.py
    student.py

  optim/
    adamw.py
    muon.py

  data/
    synthetic_data.py

  train/
    train.py
    evaluate.py
    losses.py

  experiments/
    run_sweep.py
    summarize_results.py
    make_plots.py

  results/
    small/
    medium/
    large/
```

---

# 23. Minimal Final Report Structure

When results are complete, the experiment should support a report section with:

# Method
- teacher-student synthetic autoregressive setup
- soft-target loss
- small/medium/large model tiers
- optimizer tuning protocol
- seed/repeat protocol

# Results
- convergence curves
- final loss tables
- stability observations
- scaling comparison

# Discussion
- where Muon helps
- where AdamW remains competitive or better
- whether benefits increase with size
- limitations of synthetic setup

# 24. Bottom-Line Instruction to the Agent
mplement a strictly controlled benchmark comparing AdamW-only and Muon-only on a soft-target teacher-student autoregressive bias-free MLP. Run the benchmark at small, medium, and large model scales. Use fair tuning, multiple seeds, fixed teachers, fixed eval/test sets, and report convergence, final performance, stability, and scaling behavior.

The experiment should be simple enough to implement reliably, but rigorous enough that the conclusions are meaningful.