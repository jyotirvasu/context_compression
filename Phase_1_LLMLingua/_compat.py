"""
Small transformers-version compatibility shims.

transformers >= 4.56 renamed the `torch_dtype` argument of
`from_pretrained(...)` to `dtype` and emits a deprecation warning when the old
name is used. Older versions only understand `torch_dtype`. This helper picks
the right keyword for the installed version so model loading stays quiet on new
versions while remaining compatible with old ones.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _supports_dtype_kwarg() -> bool:
    """True if AutoModel.from_pretrained accepts the new `dtype` keyword."""
    try:
        import inspect

        from transformers import AutoModelForCausalLM

        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        params = sig.parameters
        if "dtype" in params:
            return True
        # from_pretrained usually accepts **kwargs; fall back to a version check.
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            import transformers

            parts = transformers.__version__.split(".")
            major, minor = int(parts[0]), int(parts[1])
            return (major, minor) >= (4, 56)
    except Exception:
        pass
    return False


def dtype_kwarg(dtype) -> dict:
    """Return the dtype kwarg dict using the name the installed version expects."""
    key = "dtype" if _supports_dtype_kwarg() else "torch_dtype"
    return {key: dtype}


# Process-level cache of loaded models, keyed by (model_name, device). The three
# perplexity stages (context filter, sentence filter, token compressor) all use
# the SAME small LM, so loading it once and sharing it cuts a worker's resident
# memory and native-allocator churn to ~1/3 -- a major factor in the flaky
# macOS/Apple-Silicon "bus error" (SIGBUS) seen on long runs.
_MODEL_CACHE: dict = {}


def load_causal_lm(model_name: str, device: str = "cpu"):
    """Load (and cache) a causal LM + tokenizer, shared across stages in-process.

    Returns (model, tokenizer). The model is put in eval mode with float32
    weights. Subsequent calls with the same (model_name, device) return the
    exact same objects instead of allocating another copy.
    """
    key = (model_name, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **dtype_kwarg(torch.float32)
    ).to(device)
    model.eval()

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer
