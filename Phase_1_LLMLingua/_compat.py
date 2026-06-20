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
