import torch
from torch import nn

__all__ = [
    "PartialRotaryEmbedding",
]


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
