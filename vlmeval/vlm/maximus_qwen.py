"""Maximus Qwen2.5-VL adapters for VLMEvalKit.

The project pruning implementations live outside VLMEvalKit.  These adapters
bridge their ``generate`` interfaces to VLMEvalKit's ``BaseModel`` API while
keeping all selection settings explicit in the model configuration.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoProcessor

from .base import BaseModel
from .qwen2_vl.prompt import Qwen2VLPromptMixin


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "weights" / "Qwen" / "Qwen2.5-VL-7B-Instruct"


def _dtype(value: object) -> torch.dtype:
    aliases = {
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    if isinstance(value, torch.dtype):
        return value
    try:
        return aliases[str(value or "bf16").lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {value!r}; use fp16, bf16, or fp32.") from error


class _MaximusQwen(Qwen2VLPromptMixin, BaseModel):
    """Common VLMEvalKit wrapper for one Maximus Qwen selection method."""

    model_class = None
    configure_model = None
    load_config = None
    default_selection_config = None
    selection_attribute = "selection_config"
    extra_generate_kind = None

    INTERLEAVE = True

    def __init__(
        self,
        model_path: str = str(DEFAULT_MODEL_PATH),
        selection_config: str | None = None,
        device: str = "cuda:0",
        keep_ratio: float | None = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        dtype: object = "bf16",
        use_vllm: bool = False,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if use_vllm:
            raise ValueError("Maximus Qwen adapters do not support --use-vllm.")
        if kwargs:
            raise TypeError(f"Unexpected Maximus Qwen model arguments: {sorted(kwargs)}")
        if self.model_class is None or self.configure_model is None or self.load_config is None:
            raise TypeError("A concrete Maximus Qwen selection method is required.")

        BaseModel.__init__(self)
        self._use_custom_prompt = use_custom_prompt
        self.device = torch.device(device)
        self.model_path = str(Path(model_path).expanduser())
        config_path = Path(selection_config or self.default_selection_config).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        options = type(self).load_config(config_path)
        if keep_ratio is not None:
            options["keep_ratio"] = float(keep_ratio)

        self.model = self.model_class.from_pretrained(
            self.model_path,
            torch_dtype=_dtype(dtype),
            device_map={"": str(self.device)},
        ).eval()
        selection = type(self).configure_model(self.model, options)
        setattr(self, self.selection_attribute, selection)
        self.selection_config_path = str(config_path)
        self.min_pixels, self.max_pixels = int(min_pixels), int(max_pixels)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.system_prompt, self.verbose = system_prompt, verbose
        self.generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": top_p,
            "num_beams": int(num_beams),
        }

    def _prepare_content(self, message):
        content = []
        for item in message:
            if item["type"] == "image":
                content.append({
                    "type": "image", "image": item["value"],
                    "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                })
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            else:
                raise ValueError(f"Maximus Qwen supports image/text inputs only, got {item['type']!r}.")
        return content

    @staticmethod
    def _load_images(message):
        from PIL import Image

        return [Image.open(item["value"]).convert("RGB") for item in message if item["type"] == "image"]

    @staticmethod
    def _raw_image_tensors(images):
        import numpy as np

        return [
            torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
            for image in images
        ]

    @staticmethod
    def _question(message) -> str:
        return " ".join(
            str(item["value"]).replace("<image>", " ").strip()
            for item in message if item["type"] == "text"
        ).strip()

    def _cdpruner_text_mask(self, input_ids, attention_mask, prompt, question):
        """Keep only the user question as the non-CLIP CDPruner query."""
        if not question:
            raise ValueError("CDPruner requires a textual user question.")
        encoded = self.processor.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        start_char = prompt.rfind(question)
        if start_char < 0:
            raise ValueError("Could not locate the CDPruner question in the rendered Qwen prompt.")
        end_char = start_char + len(question)
        question_ids = [
            token for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"])
            if end > start_char and start < end_char
        ]
        if not question_ids:
            raise ValueError("CDPruner question did not produce tokenizer tokens.")
        target = input_ids.new_tensor(question_ids)
        active = attention_mask[0].bool()
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for offset in range(input_ids.shape[1] - target.numel() + 1):
            end = offset + target.numel()
            if bool(active[offset:end].all()) and torch.equal(input_ids[0, offset:end], target):
                mask[0, offset:end] = True
        if not bool(mask.any()):
            raise ValueError("Could not align the CDPruner question with Qwen input_ids.")
        return mask

    def generate_inner(self, message, dataset=None):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content(message)})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = self._load_images(message)
        inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors="pt").to(self.device)
        model_inputs = dict(inputs)
        input_ids = model_inputs.pop("input_ids")
        question = self._question(message)
        extra = {}

        if self.extra_generate_kind == "t2v":
            selection = getattr(self, self.selection_attribute)
            dense_method = getattr(selection, "dense_method", None)
            extra = {
                "t2v_token_ids": [" ".join(question.split())],
                "t2v_images": self._raw_image_tensors(images)
                if dense_method in {"internal", "maskclip", "sclip", "clearclip"} else None,
            }
        elif self.extra_generate_kind == "cdpruner":
            selection = getattr(self, self.selection_attribute)
            if selection.get("dense_method") is None:
                extra = {"cdpruner_text_mask": self._cdpruner_text_mask(
                    input_ids, model_inputs["attention_mask"], prompt, question
                )}
            else:
                extra = {
                    "cdpruner_images": self._raw_image_tensors(images),
                    "cdpruner_queries": [" ".join(question.split())],
                }
        elif self.extra_generate_kind == "sfpruner":
            selection = getattr(self, self.selection_attribute)
            if selection.get("dense_method") is not None:
                extra = {
                    "sfpruner_images": self._raw_image_tensors(images),
                    "sfpruner_queries": [" ".join(question.split())],
                }

        generated = self.model.generate(
            inputs=input_ids,
            **model_inputs,
            **extra,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            do_sample=self.generate_kwargs["temperature"] > 0,
            temperature=self.generate_kwargs["temperature"] or None,
            top_p=self.generate_kwargs["top_p"],
            num_beams=self.generate_kwargs["num_beams"],
            max_new_tokens=self.generate_kwargs["max_new_tokens"],
            use_cache=True,
        )
        # These selection methods pass compact ``inputs_embeds`` to ``generate``.
        # In that code path GenerationMixin returns answer tokens only; slicing
        # them by the *unpruned* input_ids length turns their answers
        # into an empty string.  Standard Qwen-style paths still return the
        # prompt followed by the answer and therefore need the usual slice.
        if self.extra_generate_kind not in {"t2v", "sfpruner", "divprune", "cdpruner", "random"}:
            generated = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class Qwen25T2V(_MaximusQwen):
    default_selection_config = str(PROJECT_ROOT / "config" / "qwen.yaml")
    selection_attribute = "t2v_config"
    extra_generate_kind = "t2v"

    from models.qwen.qwen_arch import Qwen25T2VForConditionalGeneration as model_class
    from selector.t2v.runtime import configure_t2v_model as configure_model
    from utils.run.config_file_loader import load_selection_option as load_config


class Qwen25DivPrune(_MaximusQwen):
    default_selection_config = str(PROJECT_ROOT / "config" / "divprune.yaml")
    extra_generate_kind = "divprune"

    from models.qwen.qwen_divprune_arch import Qwen25DivPruneForConditionalGeneration as model_class
    from models.qwen.qwen_divprune_arch import configure_divprune_model as configure_model
    from models.qwen.qwen_divprune_arch import load_divprune_config as load_config


class Qwen25CDPruner(_MaximusQwen):
    default_selection_config = str(PROJECT_ROOT / "config" / "cdpruner.yaml")
    selection_attribute = "cdpruner_config"
    extra_generate_kind = "cdpruner"

    from models.qwen.qwen_cdpruner_arch import Qwen25CDPrunerForConditionalGeneration as model_class
    from models.qwen.qwen_cdpruner_arch import configure_cdpruner_model as configure_model
    from models.qwen.qwen_cdpruner_arch import load_cdpruner_config as load_config


class Qwen25SFPruner(_MaximusQwen):
    default_selection_config = str(PROJECT_ROOT / "config" / "sfpruner.yaml")
    selection_attribute = "sfpruner_config"
    extra_generate_kind = "sfpruner"

    from models.qwen.qwen_sfpruner_arch import Qwen25SFPrunerForConditionalGeneration as model_class
    from models.qwen.qwen_sfpruner_arch import configure_sfpruner_model as configure_model
    from models.qwen.qwen_sfpruner_arch import load_sfpruner_config as load_config


class Qwen25Random(_MaximusQwen):
    default_selection_config = str(PROJECT_ROOT / "config" / "random.yaml")
    extra_generate_kind = "random"

    from models.qwen.qwen_random_arch import Qwen25RandomForConditionalGeneration as model_class
    from models.qwen.qwen_random_arch import configure_random_model as configure_model
    from models.qwen.qwen_random_arch import load_random_config as load_config
