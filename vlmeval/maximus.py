"""Maximus-specific registration for the vendored VLMEvalKit runner.

This module is imported explicitly by :mod:`run` before its argument parser
and model registry are loaded.  It deliberately does not rely on Python's
``sitecustomize`` import hook, so ``python VLMEvalKit/run.py`` works without a
project-root ``sitecustomize.py`` or a custom ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN_MODEL_PATH = PROJECT_ROOT / "weights" / "Qwen" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_LLAVA_NEXT_MODEL_PATH = PROJECT_ROOT / "weights" / "liuhaotian" / "llava-v1.6-vicuna-7b"


def _pop_option(argv: list[str], name: str) -> str | None:
    """Remove one CLI option and return its last supplied value."""
    value = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[index + 1]
            del argv[index:index + 2]
            continue
        prefix = f"{name}="
        if token.startswith(prefix):
            value = token[len(prefix):]
            del argv[index]
            continue
        index += 1
    return value


def register_maximus_models(argv: list[str] | None = None) -> None:
    """Register local Qwen and LLaVA-NeXT adapters and consume their CLI options.

    ``--selection-config`` is model-agnostic.  ``--t2v-config`` remains an
    alias so the existing evaluation launchers continue to work.
    """
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    argv = sys.argv if argv is None else argv
    selection_config = _pop_option(argv, "--selection-config")
    legacy_config = _pop_option(argv, "--t2v-config")
    if selection_config is not None and legacy_config is not None:
        raise SystemExit("Use only one of --selection-config and --t2v-config")
    selection_config = selection_config or legacy_config
    device = _pop_option(argv, "--device")
    keep_ratio = _pop_option(argv, "--keep-ratio")
    model_path = _pop_option(argv, "--model-path")
    dtype = _pop_option(argv, "--dtype")

    import vlmeval.vlm as vlm
    from vlmeval.config import supported_VLM
    from vlmeval.vlm import maximus_llava, maximus_qwen

    qwen_kwargs = {"model_path": str(DEFAULT_QWEN_MODEL_PATH)}
    llava_kwargs = {"model_path": str(DEFAULT_LLAVA_NEXT_MODEL_PATH)}
    if model_path is not None:
        qwen_kwargs["model_path"] = model_path
        llava_kwargs["model_path"] = model_path
    if dtype is not None:
        qwen_kwargs["dtype"] = dtype
        llava_kwargs["dtype"] = dtype
    if selection_config is not None:
        qwen_kwargs["selection_config"] = selection_config
        llava_kwargs["selection_config"] = selection_config
    if device is not None:
        qwen_kwargs["device"] = device
        llava_kwargs["device"] = device
    if keep_ratio is not None:
        try:
            value = float(keep_ratio)
        except ValueError as error:
            raise SystemExit("--keep-ratio must be a number") from error

        qwen_kwargs["keep_ratio"] = value
        llava_kwargs["keep_ratio"] = value
    registry = {
        "Qwen2.5-VL-7B-Instruct-T2V": maximus_qwen.Qwen25T2V,
        "Qwen2.5-VL-7B-Instruct-DivPrune": maximus_qwen.Qwen25DivPrune,
        "Qwen2.5-VL-7B-Instruct-CDPruner": maximus_qwen.Qwen25CDPruner,
        "Qwen2.5-VL-7B-Instruct-SFPruner": maximus_qwen.Qwen25SFPruner,
        "Qwen2.5-VL-7B-Instruct-Random": maximus_qwen.Qwen25Random,
    }
    for name, cls in registry.items():
        setattr(vlm, cls.__name__, cls)
        supported_VLM[name] = partial(cls, **qwen_kwargs)

    llava_cls = maximus_llava.LlavaNextT2VModel
    setattr(vlm, llava_cls.__name__, llava_cls)
    supported_VLM["LLaVA-NeXT-v1.6-T2V"] = partial(llava_cls, **llava_kwargs)


def register_maximus_qwen(argv: list[str] | None = None) -> None:
    """Backward-compatible alias for the local model registration hook."""
    register_maximus_models(argv)
