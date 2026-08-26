import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("kimi_delta_attention_core", mutates_args=("state",))
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    f: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    conv_kernel_size: int,
) -> torch.Tensor:
    """Return KDA output metadata while preserving the real op's state mutation schema.

    TensorCast operators are meta-only placeholders. ``mutates_args`` records that
    the real KDA kernel updates its recurrent state; this function does not compute
    or materialize that update.
    """
    del q, k, f, beta, gate, state, query_lens, seq_lens, conv_kernel_size
    return torch.empty_like(v).contiguous()
