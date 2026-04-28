from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.eagle3.config import Eagle3SpeculatorConfig

__all__ = [
    "Eagle3LCSpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle3_lc")
class Eagle3LCSpeculatorConfig(Eagle3SpeculatorConfig):
    """
    Configuration for Eagle3-LC speculator with configurable RoPE for long contexts.

    Extends Eagle3SpeculatorConfig with a ``rope_method`` field selecting among four
    positional encoding strategies:

    - "full" — standard Eagle3 RoPE (baseline, no change)
    - "yarn" — static YaRN frequency-domain scaling; requires ``rope_scaling_config``
                with keys ``factor``, ``original_max_position_embeddings``,
                and optionally ``beta_fast``/``beta_slow``
    - "dynamic_yarn" — YaRN that reverts to standard RoPE for sequences shorter than
                ``original_max_position_embeddings``, eliminating the short-context
                penalty of static YaRN; requires the same ``rope_scaling_config`` as
                "yarn"
    - "llama3" — Llama-3.1 RoPE scaling; requires ``rope_scaling_config``
                      with keys ``factor``, ``low_freq_factor``,
                      ``high_freq_factor``, ``original_max_position_embeddings``
    - "partial" — Qwen3-style partial RoPE: apply rotation to only the first
                      ``rope_partial_factor`` fraction of head dimensions, leaving the
                      rest unrotated to improve length extrapolation

    :param rope_method: Which RoPE variant to use.
    :param rope_scaling_config: Scaling parameters for "yarn" or "llama3" methods.
    :param rope_partial_factor: Fraction of head dimensions to rotate (0 < f ≤ 1).
    """

    speculators_model_type: Literal["eagle3_lc"] = "eagle3_lc"
    architectures: list[str] = Field(
        default_factory=lambda: ["Eagle3LCSpeculator"],
        description="Model architectures that can load these weights",
    )

    rope_method: Literal["full", "yarn", "dynamic_yarn", "llama3", "partial"] = Field(
        default="full",
        description="RoPE variant to use for long-context training",
    )

    rope_scaling_config: dict | None = Field(
        default=None,
        description=(
            "Scaling parameters for yarn or llama3 rope_method. "
            "For yarn: {factor, original_max_position_embeddings, beta_fast?, beta_slow?}. "
            "For llama3: {factor, low_freq_factor, high_freq_factor, original_max_position_embeddings}."
        ),
    )

    rope_partial_factor: float = Field(
        default=0.25,
        description=(
            "Fraction of head dimensions to apply RoPE to when rope_method='partial'. "
            "Remaining dimensions carry no positional information (Qwen3 style)."
        ),
    )
