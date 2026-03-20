import cupy as cp

_CANONICAL_ALIASES = {
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "single": "float32",
    "fp64": "float64",
    "float64": "float64",
    "double": "float64",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
}


def cupy_has_bfloat16() -> bool:
    if getattr(cp, "bfloat16", None) is not None:
        return True
    try:
        cp.dtype("bfloat16")
        return True
    except Exception:
        return False


def supported_dtype_names() -> list[str]:
    names = ["float16", "float32", "float64"]
    if cupy_has_bfloat16():
        names.append("bfloat16")
    return names


def normalize_dtype_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"dtype name must be a string, got {type(name)}")

    # Normalize common aliases like fp32, bf16, double.
    normalized = _CANONICAL_ALIASES.get(name.strip().lower())
    if normalized is None:
        raise ValueError(
            f"dtype {name!r} unsupported; expected one of {', '.join(supported_dtype_names())} "
            f"or aliases {', '.join(sorted(_CANONICAL_ALIASES))}"
        )

    if normalized == "bfloat16" and not cupy_has_bfloat16():
        raise ValueError("dtype 'bfloat16' requested, but this CuPy runtime does not expose bfloat16")

    if normalized not in supported_dtype_names():
        raise ValueError(f"dtype {normalized!r} unsupported by this CuPy runtime")

    return normalized


def resolve_dtype(name: str):
    normalized = normalize_dtype_name(name)
    # Prefer direct CuPy attribute, fallback to dtype constructor.
    dtype = getattr(cp, normalized, None)
    if dtype is None:
        try:
            dtype = cp.dtype(normalized)
        except Exception as exc:
            raise ValueError(f"dtype {normalized!r} could not be resolved from CuPy") from exc
    return dtype
