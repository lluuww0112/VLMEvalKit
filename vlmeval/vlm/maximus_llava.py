"""LLaVA-NeXT T2V adapter for the vendored VLMEvalKit runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .base import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "liuhaotian" / "llava-v1.6-vicuna-7b"
DEFAULT_SELECTION_CONFIG = PROJECT_ROOT / "config" / "T2V.yaml"


def _question(message: list[dict[str, object]], dataset: str | None = None) -> str:
    """Join VLMEvalKit text content without duplicating LLaVA's image marker."""
    text = "\n".join(str(item["value"]) for item in message if item["type"] == "text")
    for marker in ("<image>", "<Image>", "</Image>"):
        text = text.replace(marker, "")
    if dataset == "HallusionBench":
        text = f"{text}\nPlease answer yes or no."
    return text.strip()


class LlavaNextT2VModel(BaseModel):
    """Expose the repository's one-image LLaVA-NeXT T2V runtime to VLMEvalKit."""

    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = str(DEFAULT_MODEL_PATH),
        selection_config: str = str(DEFAULT_SELECTION_CONFIG),
        device: str = "cuda:0",
        dtype: object = "bf16",
        keep_ratio: float | None = None,
        device_map: str | None = None,
        attn_implementation: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        num_beams: int = 1,
        **kwargs,
    ) -> None:
        if kwargs:
            raise TypeError(f"Unexpected LLaVA-NeXT T2V model arguments: {sorted(kwargs)}")
        super().__init__()

        from models.llava.llava_next_t2v import LlavaNextT2V, _resolve_dtype, load_t2v_config

        config = load_t2v_config(selection_config)
        if keep_ratio is not None:
            config = replace(config, keep_ratio=float(keep_ratio)).validate()
        resolved_model_path = Path(model_path).expanduser()
        self.runner = LlavaNextT2V(
            str(resolved_model_path) if resolved_model_path.exists() else model_path,
            selection_config=config,
            device=device,
            dtype=_resolve_dtype(dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.num_beams = int(num_beams)

    def generate_inner(self, message, dataset=None):
        from PIL import Image

        image_paths = [item["value"] for item in message if item["type"] == "image"]
        if len(image_paths) != 1:
            raise ValueError("LLaVA-NeXT T2V requires exactly one image per VLMEvalKit request.")
        with Image.open(image_paths[0]) as source:
            image = source.convert("RGB")
        return self.runner.generate(
            image,
            _question(message, dataset),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            num_beams=self.num_beams,
        )


__all__ = ["LlavaNextT2VModel"]
