# SPDX-License-Identifier: Apache-2.0
"""Gluon sparse MLA kernels for the ``triton_gluon`` DSA backend.

MI355X (gfx950) only. The kernels cover the GLM-5.2 absorbed-MLA geometry:
8 or 16 local query heads, 576-wide FP8 KV rows (512 latent + 64 RoPE) and
top-k 2048. Each kernel file is tuned for one row count and is adapted from
OpenAI-Partners/artemis-kernel-integrations@14eb5a6. Row counts without a
kernel, and KV pools a kernel cannot address, run the Triton kernels.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_KV_DIM = 576
_V_DIM = 512
_TOPK = 2048
# Smallest KV-cache byte bound asserted by the kernels that use 32-bit offsets.
_INT32_KV_BYTES = 2**31 - 1024

# (rows, module, needs_int32_kv) per local head count, ascending rows.
_DECODE_KERNELS = {
    16: (
        (1, "h16_m1", False),
        (2, "h16_m2", False),
        (4, "h16_m4", True),
        (8, "h16_m8", False),
        (16, "h16_m16", False),
        (32, "h16_m32", True),
        (64, "h16_m64", True),
        (128, "h16_m128", True),
    ),
    8: (
        (1, "h8_m1", False),
        (2, "h8_m2", False),
        (4, "h8_m4", False),
        (8, "h8_m8", True),
        (16, "h8_m16", False),
        (32, "h8_m32", True),
        (64, "h8_m64", True),
        (128, "h8_m128", True),
        (256, "h8_m256", True),
    ),
}
# Prefill row slices, largest first. Rows left below the smallest slice run Triton.
_PREFILL_KERNELS = {
    16: (
        (16384, "h16_m16384", True),
        (4096, "h16_m4096", True),
        (2048, "h16_m2048", True),
        (1024, "h16_m1024", True),
    ),
    8: (
        (16384, "h8_m4193_16384", True),
        (8192, "h8_m4193_16384", True),
        (4096, "h8_m1024_4192", True),
        (2048, "h8_m1024_4192", True),
        (1024, "h8_m1024_4192", True),
    ),
}


@functools.lru_cache(maxsize=1)
def gluon_sparse_mla_unsupported_reason() -> Optional[str]:
    """Why the current runtime cannot run these kernels, or None if it can."""
    if not torch.version.hip or not torch.cuda.is_available():
        return "it requires ROCm and an AMD GPU"
    arch = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = arch.gcnArchName.split(":")[0]
    if arch != "gfx950":
        return f"it requires an MI355X (gfx950) GPU, found {arch}"
    try:
        import triton
        import triton.experimental.gluon.language as gl
    except ImportError as e:
        return f"Triton Gluon is unavailable ({e})"
    from packaging.version import Version

    if Version(Version(triton.__version__).base_version) < Version("3.8.0"):
        return f"it requires Triton >= 3.8.0, found Triton {triton.__version__}"
    if not hasattr(getattr(gl, "amd", None), "cdna4"):
        return f"Triton {triton.__version__} lacks the Gluon CDNA4 API"
    return None


def gluon_sparse_mla_contract_error(
    *,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    topk: int,
    kv_cache_dtype: torch.dtype,
) -> Optional[str]:
    """Why a model/configuration does not match the kernels, or None."""
    if num_heads not in _DECODE_KERNELS:
        return f"it supports 8 or 16 query heads per rank, got {num_heads}"
    if kv_lora_rank + qk_rope_head_dim != _KV_DIM or kv_lora_rank != _V_DIM:
        return (
            f"it requires kv_lora_rank={_V_DIM} and qk_rope_head_dim="
            f"{_KV_DIM - _V_DIM}, got {kv_lora_rank} and {qk_rope_head_dim}"
        )
    if topk != _TOPK:
        return f"it requires index_topk={_TOPK}, got {topk}"
    if kv_cache_dtype != torch.float8_e4m3fn:
        return f"it requires an fp8_e4m3 KV cache, got {kv_cache_dtype}"
    return None


@functools.lru_cache(maxsize=None)
def _kernel(module: str) -> Callable[..., torch.Tensor]:
    return importlib.import_module(f"{__name__}.{module}").sparse_paged_mla


@functools.lru_cache(maxsize=None)
def _log_large_kv(heads: int) -> None:
    logger.info(
        "triton_gluon: the KV cache exceeds 2 GiB per layer; kernels with 32-bit "
        "KV offsets are skipped (%d heads) and those row counts run Triton.",
        heads,
    )


@functools.lru_cache(maxsize=None)
def _decode_choice(heads: int, rows: int, large_kv: bool) -> Optional[tuple[int, str]]:
    for bucket, module, needs_int32_kv in _DECODE_KERNELS[heads]:
        if bucket >= rows and not (large_kv and needs_int32_kv):
            return bucket, module
    return None


@functools.lru_cache(maxsize=None)
def _prefill_choice(heads: int, large_kv: bool) -> tuple[tuple[int, str], ...]:
    return tuple(
        (size, module)
        for size, module, needs_int32_kv in _PREFILL_KERNELS[heads]
        if not (large_kv and needs_int32_kv)
    )


def _joined_query(q_nope: torch.Tensor, q_rope: torch.Tensor) -> torch.Tensor:
    """[N, H, 576] BF16 query; zero-copy when q_nope/q_rope split one buffer."""
    rows, heads, d_v = q_nope.shape
    if (
        q_rope.data_ptr() == q_nope.data_ptr() + d_v * q_nope.element_size()
        and q_rope.stride() == q_nope.stride()
        and q_nope.stride(-1) == 1
        and q_nope.stride(1) >= _KV_DIM
    ):
        query = q_nope.as_strided((rows, heads, _KV_DIM), q_nope.stride())
    else:
        query = torch.cat([q_nope, q_rope], dim=-1)
    return query if query.dtype == torch.bfloat16 else query.to(torch.bfloat16)


def _gluon_sparse_mla(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> Optional[torch.Tensor]:
    """[1, N, H, d_v] BF16, or None when no Gluon kernel covers the call."""
    rows, heads, _ = q_nope.shape
    if (
        rows == 0
        or d_v != _V_DIM
        or heads not in _DECODE_KERNELS
        or q_rope.shape[-1] != _KV_DIM - _V_DIM
        or kv.dtype != torch.float8_e4m3fn
        or kv.shape[-1] != _KV_DIM
        or indices.shape[-1] != _TOPK
    ):
        return None
    kv_2d = kv.view(-1, _KV_DIM)
    large_kv = (kv_2d.shape[0] - 1) * kv_2d.stride(0) + _KV_DIM >= _INT32_KV_BYTES
    if large_kv:
        _log_large_kv(heads)
    slots = indices.reshape(rows, _TOPK)
    if slots.dtype != torch.int32 or slots.stride(-1) != 1:
        slots = slots.to(torch.int32).contiguous()

    if rows <= _DECODE_KERNELS[heads][-1][0]:
        choice = _decode_choice(heads, rows, large_kv)
        if choice is None:
            return None
        bucket, module = choice
        query = _joined_query(q_nope, q_rope)
        if bucket > rows:
            query = F.pad(query, (0, 0, 0, 0, 0, bucket - rows))
            slots = F.pad(slots, (0, 0, 0, bucket - rows), value=-1)
        out = _kernel(module)(query, kv_2d, slots, softmax_scale=sm_scale)
        return out[:rows].unsqueeze(0)

    slices = _prefill_choice(heads, large_kv)
    if not slices or rows < slices[-1][0]:
        return None
    query = _joined_query(q_nope, q_rope)
    out = query.new_empty((rows, heads, _V_DIM))
    start = 0
    for size, module in slices:
        while rows - start >= size:
            end = start + size
            _kernel(module)(
                query[start:end],
                kv_2d,
                slots[start:end],
                softmax_scale=sm_scale,
                output=out[start:end],
            )
            start = end
    if start < rows:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd,
        )

        out[start:] = triton_sparse_mla_fwd(
            q_nope[start:], q_rope[start:], kv, indices[start:], sm_scale, d_v
        )[0]
    return out.unsqueeze(0)


def gluon_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Drop-in for ``triton_sparse_mla_fwd``: prefill, target verify, draft extend."""
    out = _gluon_sparse_mla(q_nope, q_rope, kv, indices, sm_scale, d_v)
    if out is None:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
            triton_sparse_mla_fwd,
        )

        return triton_sparse_mla_fwd(q_nope, q_rope, kv, indices, sm_scale, d_v)
    return out


def gluon_sparse_mla_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Drop-in for ``triton_sparse_mla_decode_splitk``."""
    out = _gluon_sparse_mla(q_nope, q_rope, kv, indices, sm_scale, d_v)
    if out is None:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_splitk,
        )

        return triton_sparse_mla_decode_splitk(
            q_nope, q_rope, kv, indices, sm_scale, d_v
        )
    return out
