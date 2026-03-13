import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from muon_analysis.dtypes import normalize_dtype_name


def _resolve_json_path(path) -> Path:
    candidate = Path(path)
    search_order = [candidate]

    if not candidate.is_absolute():
        search_order.append(Path("configs") / candidate)
        if candidate.suffix != ".json":
            search_order.append(candidate.with_suffix(".json"))
            search_order.append(Path("configs") / candidate.with_suffix(".json"))

    for resolved in search_order:
        if resolved.exists():
            return resolved

    return candidate

def _filtered_kwargs(cls, data: dict) -> dict:
    field_names = cls.__dataclass_fields__.keys()
    return {key: value for key, value in data.items() if key in field_names}


@dataclass
class TuningConfig:
    learning_rates: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3])
    weight_decays: list[float] = field(default_factory=lambda: [0.0, 1e-4, 5e-4])

    def validate(self):
        assert self.learning_rates, "learning_rates must not be empty"
        assert self.weight_decays, "weight_decays must not be empty"
        for value in self.learning_rates:
            assert value > 0.0, f"learning rate must be positive, got {value}"
        for value in self.weight_decays:
            assert value >= 0.0, f"weight decay must be non-negative, got {value}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None):
        return cls(**_filtered_kwargs(cls, data or {}))


@dataclass
class Config:
    V: int
    K: int
    T: int
    width: int
    layers: int
    activation: str = "relu"
    dtype: str = "float32"
    temperature: float = 1.0
    prefix_mode: str = "bos"
    bos_id: int = 0
    teacher_seed: int = 0
    validation_seed: int = 1
    test_seed: int = 2
    train_seed: int = 3
    student_seed: int = 4
    B_train: int = 32
    B_val: int = 128
    B_test: int = 128
    train_data_mode: str = "precomputed"
    train_batch_block_size: int = 64
    steps: int = 100
    eval_every: int = 10
    log_every: int = 250
    finite_check_every: int = 100
    step_log_flush_every: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    muon_ns_steps: int = 3
    num_threads: int = 2
    run_name: str = "default"
    tier_name: str = "default"

    def __post_init__(self):
        self.dtype = normalize_dtype_name(self.dtype)
        self.validate()

    def validate(self):
        assert self.T > self.K, f"Need T > K, got T={self.T}, K={self.K}"
        assert self.V > 0 and self.K > 0 and self.width > 0 and self.layers > 0, "Model parameters invalid"
        assert self.B_train > 0 and self.B_val > 0 and self.B_test > 0, "Batch sizes must be positive"
        assert self.train_data_mode in {"online", "precomputed"}, f"train_data_mode {self.train_data_mode} unsupported"
        assert self.train_batch_block_size > 0, f"train_batch_block_size must be positive, got {self.train_batch_block_size}"
        assert self.steps > 0 and self.eval_every > 0, "Training schedule invalid"
        assert self.log_every > 0, f"log_every must be positive, got {self.log_every}"
        assert self.finite_check_every > 0, f"finite_check_every must be positive, got {self.finite_check_every}"
        assert self.step_log_flush_every > 0, f"step_log_flush_every must be positive, got {self.step_log_flush_every}"
        assert self.temperature > 0.0, f"temperature must be positive, got {self.temperature}"
        assert self.activation in {"relu", "gelu"}, f"activation {self.activation} unsupported"
        assert self.prefix_mode in {"bos", "random"}, f"prefix_mode {self.prefix_mode} unsupported"
        assert self.lr > 0.0, f"lr must be positive, got {self.lr}"
        assert self.weight_decay >= 0.0, f"weight_decay must be non-negative, got {self.weight_decay}"
        assert self.optimizer in {"adamw", "muon", "muon_hybrid"}, f"optimizer {self.optimizer} unsupported"
        assert self.muon_ns_steps > 0, f"muon_ns_steps must be positive, got {self.muon_ns_steps}"
        assert self.num_threads >= 1, f"num_threads must be at least 1, got {self.num_threads}"
        assert 0 <= self.bos_id < self.V, f"bos_id {self.bos_id} must be in [0, V)"

    @property
    def in_dim(self) -> int:
        return self.K * self.V

    def clone(self, **overrides):
        return replace(self, **overrides)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict):
        payload = dict(data)

        if "B_eval" in payload:
            payload.setdefault("B_val", payload["B_eval"])
            payload.setdefault("B_test", payload["B_eval"])
        if "eval_seed" in payload:
            payload.setdefault("validation_seed", payload["eval_seed"])
            payload.setdefault("test_seed", payload["eval_seed"] + 1)

        if "optimizers" in payload and "optimizer" not in payload and payload["optimizers"]:
            payload["optimizer"] = payload["optimizers"][0]

        return cls(**_filtered_kwargs(cls, payload))

    @classmethod
    def load_json(cls, path):
        with _resolve_json_path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)


@dataclass
class TierConfig:
    name: str
    V: int
    K: int
    T: int
    width: int
    layers: int
    activation: str = "relu"
    dtype: str = "float32"
    temperature: float = 1.0
    prefix_mode: str = "bos"
    bos_id: int = 0
    teacher_seed: int = 0
    validation_seed: int = 1
    test_seed: int = 2
    student_seeds: list[int] = field(default_factory=lambda: [11, 12, 13])
    train_seeds: list[int] = field(default_factory=lambda: [21, 22, 23])
    B_train: int = 32
    B_val: int = 128
    B_test: int = 128
    train_data_mode: str = "precomputed"
    train_batch_block_size: int = 64
    steps: int = 100
    eval_every: int = 10
    log_every: int = 250
    finite_check_every: int = 100
    step_log_flush_every: int = 50
    muon_ns_steps: int = 3
    num_threads: int = 2
    tuning: TuningConfig | None = None

    def __post_init__(self):
        self.dtype = normalize_dtype_name(self.dtype)
        if self.tuning is not None and not isinstance(self.tuning, TuningConfig):
            self.tuning = TuningConfig.from_dict(self.tuning)
        self.validate()

    def validate(self):
        assert self.name, "tier name must not be empty"
        assert self.T > self.K, f"Need T > K, got T={self.T}, K={self.K}"
        assert self.V > 0 and self.K > 0 and self.width > 0 and self.layers > 0, "Model parameters invalid"
        assert self.B_train > 0 and self.B_val > 0 and self.B_test > 0, "Batch sizes must be positive"
        assert self.train_data_mode in {"online", "precomputed"}, f"train_data_mode {self.train_data_mode} unsupported"
        assert self.train_batch_block_size > 0, f"train_batch_block_size must be positive, got {self.train_batch_block_size}"
        assert self.steps > 0 and self.eval_every > 0, "Training schedule invalid"
        assert self.log_every > 0, f"log_every must be positive, got {self.log_every}"
        assert self.finite_check_every > 0, f"finite_check_every must be positive, got {self.finite_check_every}"
        assert self.step_log_flush_every > 0, f"step_log_flush_every must be positive, got {self.step_log_flush_every}"
        assert self.temperature > 0.0, f"temperature must be positive, got {self.temperature}"
        assert self.activation in {"relu", "gelu"}, f"activation {self.activation} unsupported"
        assert self.prefix_mode in {"bos", "random"}, f"prefix_mode {self.prefix_mode} unsupported"
        assert self.muon_ns_steps > 0, f"muon_ns_steps must be positive, got {self.muon_ns_steps}"
        assert self.num_threads >= 1, f"num_threads must be at least 1, got {self.num_threads}"
        assert 0 <= self.bos_id < self.V, f"bos_id {self.bos_id} must be in [0, V)"
        assert self.student_seeds, "student_seeds must not be empty"
        assert self.train_seeds, "train_seeds must not be empty"
        assert len(self.student_seeds) == len(self.train_seeds), "student_seeds and train_seeds must have the same length"
        if self.tuning is not None:
            self.tuning.validate()

    @property
    def repeats(self) -> int:
        return len(self.student_seeds)

    def to_run_config(self, *, benchmark_name: str, optimizer: str, lr: float, weight_decay: float, repeat_index: int) -> Config:
        return Config(
            V=self.V,
            K=self.K,
            T=self.T,
            width=self.width,
            layers=self.layers,
            activation=self.activation,
            dtype=self.dtype,
            temperature=self.temperature,
            prefix_mode=self.prefix_mode,
            bos_id=self.bos_id,
            teacher_seed=self.teacher_seed,
            validation_seed=self.validation_seed,
            test_seed=self.test_seed,
            train_seed=self.train_seeds[repeat_index],
            student_seed=self.student_seeds[repeat_index],
            B_train=self.B_train,
            B_val=self.B_val,
            B_test=self.B_test,
            train_data_mode=self.train_data_mode,
            train_batch_block_size=self.train_batch_block_size,
            steps=self.steps,
            eval_every=self.eval_every,
            log_every=self.log_every,
            finite_check_every=self.finite_check_every,
            step_log_flush_every=self.step_log_flush_every,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            muon_ns_steps=self.muon_ns_steps,
            num_threads=self.num_threads,
            run_name=benchmark_name,
            tier_name=self.name,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.tuning is not None:
            payload["tuning"] = self.tuning.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict):
        payload = dict(data)
        if "B_eval" in payload:
            payload.setdefault("B_val", payload["B_eval"])
            payload.setdefault("B_test", payload["B_eval"])
        if "eval_seed" in payload:
            payload.setdefault("validation_seed", payload["eval_seed"])
            payload.setdefault("test_seed", payload["eval_seed"] + 1)
        return cls(**_filtered_kwargs(cls, payload))


@dataclass
class BenchmarkConfig:
    run_name: str
    optimizers: list[str] = field(default_factory=lambda: ["adamw", "muon"])
    tuning: TuningConfig = field(default_factory=TuningConfig)
    tiers: list[TierConfig] = field(default_factory=list)
    output_root: str = "results"
    max_parallel_trials: int = 1

    def __post_init__(self):
        if not isinstance(self.tuning, TuningConfig):
            self.tuning = TuningConfig.from_dict(self.tuning)
        self.tiers = [tier if isinstance(tier, TierConfig) else TierConfig.from_dict(tier) for tier in self.tiers]
        self.validate()

    def validate(self):
        assert self.run_name, "run_name must not be empty"
        assert self.optimizers, "optimizers must not be empty"
        assert self.max_parallel_trials >= 1, f"max_parallel_trials must be at least 1, got {self.max_parallel_trials}"
        for name in self.optimizers:
            assert name in {"adamw", "muon", "muon_hybrid"}, f"optimizer {name} unsupported"
        assert self.tiers, "tiers must not be empty"
        self.tuning.validate()
        seen = set()
        for tier in self.tiers:
            tier.validate()
            assert tier.name not in seen, f"duplicate tier name {tier.name}"
            seen.add(tier.name)

    def tier_tuning(self, tier: TierConfig) -> TuningConfig:
        return tier.tuning if tier.tuning is not None else self.tuning

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "optimizers": self.optimizers,
            "tuning": self.tuning.to_dict(),
            "tiers": [tier.to_dict() for tier in self.tiers],
            "output_root": self.output_root,
            "max_parallel_trials": self.max_parallel_trials,
        }

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**_filtered_kwargs(cls, data))

    @classmethod
    def load_json(cls, path):
        with _resolve_json_path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)