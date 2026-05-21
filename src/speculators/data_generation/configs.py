"""Configuration registries for data generation pipeline."""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "DATASET_CONFIGS",
    "DatasetConfig",
]


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset"""

    name: str
    hf_path: str
    split: str
    normalize_fn: Callable[[dict], dict] | None = None


def _normalize_ultrachat(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


def _normalize_longalpaca(example: dict) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": example.get("instruction", "")},
            {"role": "assistant", "content": example.get("output", "")},
        ]
    }


def _normalize_longalign(example: dict) -> dict:
    if "messages" in example:
        return {"conversations": example["messages"]}
    return example


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sharegpt": DatasetConfig(
        name="sharegpt",
        hf_path="Aeala/ShareGPT_Vicuna_unfiltered",
        split="train",
    ),
    "ultrachat": DatasetConfig(
        name="ultrachat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        normalize_fn=_normalize_ultrachat,
    ),
    "longalpaca": DatasetConfig(
        name="longalpaca",
        hf_path="Yukang/LongAlpaca-16k-length",
        split="train",
        normalize_fn=_normalize_longalpaca,
    ),
    "longalign": DatasetConfig(
        name="longalign",
        hf_path="THUDM/LongAlign-10k",
        split="train",
        normalize_fn=_normalize_longalign,
    ),
}
