import copy
from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.eagle3_lc.config import Eagle3LCSpeculatorConfig
from speculators.models.eagle3_lc.rotary import PartialRotaryEmbedding
from speculators.proposals.greedy import GreedyTokenProposalConfig


@SpeculatorModel.register("eagle3_lc")
class Eagle3LCDraftModel(Eagle3DraftModel):
    """Eagle3 draft model with configurable RoPE for long-context training.

    Inherits all of Eagle3DraftModel's forward pass, TTT logic, loss, and
    data-loading interfaces.  Only the rotary positional embedding setup is
    overridden via the ``rope_method`` field in ``Eagle3LCSpeculatorConfig``.

    Supported methods:

    - "full" — identical to vanilla Eagle3 (baseline)
    - "yarn" — YaRN frequency scaling; ``rope_scaling_config`` must include
                      ``{"rope_type": "yarn", "factor": ..., "original_max_position_embeddings": ...}``
    - "llama3" — Llama-3.1 scaling; ``rope_scaling_config`` must include
                      ``{"rope_type": "llama3", "factor": ..., "low_freq_factor": ...,
                      "high_freq_factor": ..., "original_max_position_embeddings": ...}``
    - "partial" — Qwen3-style partial RoPE: only the first ``rope_partial_factor``
                      fraction of head dimensions are rotated;
    """

    config_class: ClassVar[type[Eagle3LCSpeculatorConfig]] = Eagle3LCSpeculatorConfig  # type: ignore[misc,assignment]

    def _setup_rotary_embedding(self, transformer_layer_config: PretrainedConfig):
        config: Eagle3LCSpeculatorConfig = self.config  # type: ignore[assignment]

        modified_config = copy.copy(transformer_layer_config)
        modified_config.hidden_size = modified_config.hidden_size * 2

        match config.rope_method:
            case "full":
                self.rotary_emb = self._model_definitions.rotary_emb_class(
                    modified_config
                )

            case "yarn" | "llama3":
                if config.rope_scaling_config is not None:
                    modified_config = copy.copy(modified_config)
                    modified_config.rope_scaling = config.rope_scaling_config
                self.rotary_emb = self._model_definitions.rotary_emb_class(
                    modified_config
                )

            case "partial":
                base_emb = self._model_definitions.rotary_emb_class(modified_config)
                head_dim = getattr(
                    transformer_layer_config,
                    "head_dim",
                    transformer_layer_config.hidden_size
                    // transformer_layer_config.num_attention_heads,
                )
                n_rotated = int(head_dim * config.rope_partial_factor)
                n_rotated = (n_rotated // 2) * 2
                if n_rotated < 2:
                    raise ValueError(
                        f"rope_partial_factor={config.rope_partial_factor} yields "
                        f"n_rotated={n_rotated} for head_dim={head_dim}. "
                        "Must be at least 2 (one rotation pair)."
                    )
                self.rotary_emb = PartialRotaryEmbedding(base_emb, n_rotated)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "Eagle3LCDraftModel":
        """Create Eagle3LC model from training arguments.

        :param verifier_config: Verifier model configuration.
        :param kwargs: Training arguments; recognises ``rope_method``,
            ``rope_scaling_config``, and ``rope_partial_factor`` in addition to
            all kwargs accepted by ``Eagle3DraftModel.from_training_args``.
        """
        config = Eagle3LCSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            norm_before_residual=kwargs["norm_before_residual"],
            embed_requires_grad=kwargs.get("embed_requires_grad", False),
            rope_method=kwargs.get("rope_method", "full"),
            rope_scaling_config=kwargs.get("rope_scaling_config"),
            rope_partial_factor=kwargs.get("rope_partial_factor", 0.25),
            speculators_config=SpeculatorsConfig(
                algorithm="eagle3_lc",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs["ttt_steps"],
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )
        return cls(
            config=config,
            t2d=kwargs.get("t2d"),
            d2t=kwargs.get("d2t"),
        )
