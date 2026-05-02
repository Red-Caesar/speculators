import copy

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS,
    _compute_default_rope_parameters,
    _compute_yarn_parameters,
)

__all__ = [
    "DynamicYaRNRotaryEmbedding",
    "PartialRotaryEmbedding",
]


class DynamicYaRNRotaryEmbedding(LlamaRotaryEmbedding):
    """YaRN rotary embedding that applies frequency scaling only beyond the original context length.

    For sequences up to ``original_max_position_embeddings`` (taken from the YaRN
    ``rope_scaling`` config dict), the base model's own RoPE frequencies are used —
    matching what the verifier and baseline draft model use at short contexts.  For
    longer sequences the YaRN-scaled frequencies are computed on demand and cached.
    When the sequence length drops back below the original limit the frequencies are
    reset to the base values.

    This eliminates the short-context accuracy penalty of static YaRN while retaining
    its long-context extension properties.  The dynamic update logic is handled by the
    ``dynamic_rope_update`` decorator inherited from ``LlamaRotaryEmbedding.forward``;
    setting ``rope_type = "dynamic_yarn"`` satisfies the ``"dynamic" in self.rope_type``
    check inside that decorator.

    The config is expected to carry a ``_original_rope_scaling`` attribute (set by
    ``Eagle3LCDraftModel._setup_rotary_embedding``) containing the verifier model's
    own ``rope_scaling`` dict.  When present, those base frequencies are used as the
    short-context default so the draft stays aligned with the verifier.

    :param config: Model config with ``rope_scaling`` set to the YaRN parameters.
        Should also carry ``_original_rope_scaling`` with the verifier's native scaling.
    :param device: Optional device for frequency initialisation.
    """

    def __init__(self, config, device=None):
        nn.Module.__init__(self)
        self.rope_type = "dynamic_yarn"
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        original_max = rope_scaling.get(
            "original_max_position_embeddings",
            config.max_position_embeddings,
        )
        self.max_seq_len_cached = original_max
        self.original_max_seq_len = original_max
        self.config = config
        self.rope_init_fn = _compute_yarn_parameters

        original_rope_scaling = getattr(config, "_original_rope_scaling", None)
        if original_rope_scaling is not None:
            base_config = copy.copy(config)
            base_config.rope_scaling = original_rope_scaling
            base_rope_type = original_rope_scaling.get("rope_type", "default")
            base_init_fn = ROPE_INIT_FUNCTIONS.get(base_rope_type, _compute_default_rope_parameters)
            inv_freq, self.attention_scaling = base_init_fn(base_config, device)
        else:
            inv_freq, self.attention_scaling = _compute_default_rope_parameters(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.original_attention_scaling = self.attention_scaling

    @torch.no_grad()
    @torch._dynamo.disable
    def forward(self, x, position_ids):
        """Run eagerly to prevent torch.compile recompilation.

        The parent's ``@dynamic_rope_update`` decorator calls ``register_buffer``
        to swap ``inv_freq`` at runtime, which invalidates compiled subgraphs and
        causes ranks to desynchronise under FSDP. Disabling dynamo here forces the
        entire rotary embedding (including the dynamic frequency update) to run
        outside the compiled graph.

        The decorator resets ``inv_freq`` to base frequencies for short sequences but
        does not reset ``attention_scaling``.  We restore it here so that short-context
        inference after a long-context forward does not inherit the YaRN mscale.
        """
        seq_len = int(position_ids.max().item()) + 1
        if seq_len <= self.original_max_seq_len:
            self.attention_scaling = self.original_attention_scaling
        return super().forward(x, position_ids)


class PartialRotaryEmbedding(nn.Module):
    """Qwen3-style partial rotary positional embedding.

    Applies the base rotary embedding to only the first ``n_pairs`` frequency pairs
    and replaces the remaining pairs with identity values (cos=1, sin=0).  This
    leaves ``head_dim - n_rotated`` dimensions unrotated, improving length
    extrapolation without changing the model architecture.

    Background — the standard ``rotate_half`` convention pairs positions as
    ``(dim_i, dim_{i + head_dim//2})``.  Consequently ``n_rotated`` active dimensions
    correspond to ``n_pairs = n_rotated // 2`` frequency bins, and we must zero out
    *both* halves of the cos/sin arrays symmetrically:

    ::

        cos layout for head_dim=128, n_rotated=32, n_pairs=16:
          [cos[0:16]          | 1.0 * 48  | cos[64:80]          | 1.0 * 48]
          ← active (real) →  ← identity → ← active (imaginary)→ ← identity →

    The ``base_emb`` is expected to duplicate frequencies so that
    ``cos[..., i] == cos[..., head_dim//2 + i]`` (as ``LlamaRotaryEmbedding`` does).

    :param base_emb: Underlying rotary embedding module.
    :param n_rotated: Total number of head dimensions to rotate (must be even).
    """

    def __init__(self, base_emb: nn.Module, n_rotated: int):
        super().__init__()
        if n_rotated % 2 != 0:
            raise ValueError(
                f"n_rotated must be even (got {n_rotated}). "
                "Each rotation pair covers two dimensions."
            )
        self.base_emb = base_emb
        self.n_pairs = n_rotated // 2

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors with only the first n_rotated dimensions active.

        The returned tensors have the same shape as those from the underlying
        ``base_emb`` but with identity values (cos=1, sin=0) for all dimensions
        beyond ``n_rotated``.

        :param hidden_states: Used for device/dtype inference only.
        :param position_ids: Shape ``[batch, seq_len]``.
        :returns: ``(cos, sin)`` each matching the base embedding's output shape.
        """
        cos, sin = self.base_emb(hidden_states, position_ids)
        n = self.n_pairs
        h2 = cos.shape[-1] // 2

        ones = torch.ones_like(cos[..., n:h2])
        zeros = torch.zeros_like(sin[..., n:h2])

        cos_out = torch.cat([cos[..., :n], ones, cos[..., h2 : h2 + n], ones], dim=-1)
        sin_out = torch.cat([sin[..., :n], zeros, sin[..., h2 : h2 + n], zeros], dim=-1)
        return cos_out, sin_out
