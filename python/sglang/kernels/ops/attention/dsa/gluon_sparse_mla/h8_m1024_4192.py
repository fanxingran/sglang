# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m1024_4192.py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

HEADS = gl.constexpr(8)
TILE = gl.constexpr(64)
LATENT = gl.constexpr(512)
QK_CHUNK = gl.constexpr(256)
PV_CHUNK = gl.constexpr(128)
MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 2]
    )
)
LHS = gl.constexpr(gl.DotOperandLayout(0, MMA, 16))
RHS = gl.constexpr(gl.DotOperandLayout(1, MMA, 16))
COPY = gl.constexpr(gl.BlockedLayout([1, 16], [16, 4], [2, 1], [1, 0]))
QUERY = gl.constexpr(gl.BlockedLayout([1, 8], [4, 16], [2, 1], [1, 0]))
STATS = gl.constexpr(gl.BlockedLayout([1, 4], [4, 16], [2, 1], [1, 0]))


@gluon.constexpr_function
def _cache_layout(swizzle):
    return gl.SharedLinearLayout(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 32 if swizzle else 0],
            [16, 16 if swizzle else 0],
            [0, 64],
            [0, 128],
            [0, 256],
            [32, 0],
        ],
        [],
        16,
    )


@gluon.jit
def _allocate_tile_storage(dtype: gl.constexpr):
    cache = gl.allocate_shared_memory(dtype, (TILE, LATENT), _cache_layout(True))
    copy = cache._reinterpret(dtype, (TILE, LATENT), _cache_layout(False))
    rope = gl.allocate_shared_memory(
        dtype,
        (TILE, 64),
        gl.SharedLinearLayout(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 32],
                [16, 16],
                [32, 0],
            ],
            [],
            16,
        ),
    )
    rope_copy = rope._reinterpret(
        dtype, (TILE, 64), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    probability = rope._reinterpret(
        gl.bfloat16, (16, TILE * 2), gl.SwizzledSharedLayout(32, 1, 4, [1, 0])
    ).slice(0, TILE, 1)
    alpha = gl.allocate_shared_memory(
        gl.float32, (16,), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    return (cache, copy, rope, rope_copy, probability, alpha)


@gluon.jit
def _load_query(Q, row, Q0: gl.constexpr, Q1: gl.constexpr):
    head = gl.arange(0, HEADS, gl.SliceLayout(1, QUERY))
    latent = gl.arange(0, LATENT, gl.SliceLayout(0, QUERY))
    rope = gl.arange(0, 64, gl.SliceLayout(0, QUERY))
    q_latent = gl.amd.cdna4.buffer_load(
        Q + row * Q0, head[:, None] * Q1 + latent[None, :]
    )
    q_rope = gl.amd.cdna4.buffer_load(
        Q + row * Q0, head[:, None] * Q1 + LATENT + rope[None, :]
    )
    return (q_latent, q_rope)


@gluon.jit
def _stage_query(q_latent, q_rope):
    shared = gl.allocate_shared_memory(
        gl.bfloat16, (HEADS, 1024), gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    latent = shared.slice(0, LATENT, 1)
    rotary = shared.slice(LATENT, 64, 1)
    latent.store(q_latent)
    rotary.store(q_rope)
    q_latent = latent.load(LHS)
    q_rope = rotary.load(LHS)
    return (q_latent, q_rope)


@gluon.jit
def _bit_or(left, right):
    return left | right


@gluon.jit
def _first_active(bits):
    return gl.inline_asm_elementwise(
        "s_ff1_i32_b32 $0, $1", "=s,s", [bits], dtype=gl.int32, is_pure=True, pack=1
    )


@gluon.jit
def _uniform_bitmap(bits):
    return gl.inline_asm_elementwise(
        "s_mov_b32 $0, $1", "=s,s", [bits], dtype=gl.uint32, is_pure=True, pack=1
    )


@gluon.jit
def _remaining_tiles(bits):
    return gl.inline_asm_elementwise(
        "s_sub_u32 $0, $1, 1\ns_and_b32 $0, $1, $0",
        "=&s,s,~{scc}",
        [bits],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _copy_region(
    shared, KV, slot, channel, active, K0: gl.constexpr, SWIZZLE: gl.constexpr = False
):
    row_offset = gl.where(active, slot.to(gl.int32) * K0, -2147483648).to(gl.uint32)
    copy_channel = channel[None, :]
    if SWIZZLE:
        local_row = gl.arange(0, slot.shape[0], slot.type.layout)
        copy_channel = (
            copy_channel ^ (local_row[:, None] & 8) << 2 ^ local_row[:, None] & 16
        )
    offsets = row_offset[:, None] + copy_channel
    offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
    gl.amd.cdna4.async_copy.buffer_load_to_shared(shared, KV, offsets)


@gluon.jit
def _scan_tiles(S, row, S0: gl.constexpr):
    scan_layout: gl.constexpr = gl.BlockedLayout([16], [64], [2], [0])
    SCAN_SIZE: gl.constexpr = 2048
    scan_col = gl.arange(0, SCAN_SIZE, scan_layout)
    scan_slots = gl.load(S + row * S0 + scan_col)
    SIGN_BIT: gl.constexpr = S.dtype.element_ty.primitive_bitwidth - 1
    word_values = (~(scan_slots >> SIGN_BIT)).to(gl.uint32) & 1 << (scan_col % 32).to(
        gl.uint32
    )
    validity_words = gl.reduce(
        gl.reshape(word_values, (SCAN_SIZE // 32, 32)), 1, _bit_or
    )
    validity_shared = gl.allocate_shared_memory(
        gl.uint32, (SCAN_SIZE // 32,), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    validity_shared.store(validity_words)
    word_pairs = gl.reshape(validity_words, (SCAN_SIZE // 64, 2))
    active = gl.max(word_pairs, 1) != 0
    full = gl.min(word_pairs, 1) == 4294967295
    tile_id = gl.arange(0, SCAN_SIZE // 64, active.type.layout)
    bit = 1 << tile_id.to(gl.uint32)
    packed_maps = (
        gl.where(active, bit, 0).to(gl.uint64)
        | gl.where(full, bit, 0).to(gl.uint64) << 32
    )
    maps = gl.reduce(packed_maps, 0, _bit_or)
    active_tiles = _uniform_bitmap(maps.to(gl.uint32))
    full_tiles = _uniform_bitmap((maps >> 32).to(gl.uint32))
    return (active_tiles, full_tiles, validity_shared)


@gluon.jit
def _tile_scores(
    q_latent,
    q_rope,
    cache_shared,
    rope_shared,
    full_tiles,
    block,
    validity_shared,
    stats_col,
):
    score = gl.full((HEADS, TILE), 0, gl.float32, MMA)
    gl.amd.cdna4.async_copy.wait_group(0)
    q_parts = gl.split(
        gl.permute(gl.reshape(q_latent, (HEADS, 2, QK_CHUNK)), (0, 2, 1))
    )
    for key_chunk in gl.static_range(LATENT // QK_CHUNK):
        key_shared = cache_shared.permute((1, 0)).slice(
            key_chunk * QK_CHUNK, QK_CHUNK, 0
        )
        k_part = gl.amd.cdna4.async_copy.load_shared_relaxed(key_shared, RHS).to(
            gl.bfloat16
        )
        score = gl.amd.cdna4.mfma(
            gl.convert_layout(q_parts[key_chunk], LHS), k_part, score
        )
    key_rope = gl.amd.cdna4.async_copy.load_shared_relaxed(
        rope_shared.permute((1, 0)), RHS
    ).to(gl.bfloat16)
    score = gl.amd.cdna4.mfma(q_rope, key_rope, score)
    if full_tiles & 1 << block.to(gl.uint32) != 0:
        valid = gl.full((TILE,), True, gl.int1, gl.SliceLayout(0, STATS))
    else:
        words = validity_shared.gather(block * 2 + stats_col // 32, axis=0)
        valid = words >> stats_col % 32 & 1 != 0
    return (score, valid)


@gluon.jit
def _softmax_update(
    score, valid, maximum, denominator, scale, probability_shared, alpha_shared
):
    score = gl.convert_layout(score, STATS)
    score = score * scale
    score = gl.where(valid[None, :], score, -float("inf"))
    next_max = gl.maximum(maximum, gl.max(score, 1))
    alpha = gl.exp2((maximum - next_max) * 1.4426950408889634)
    probability = gl.where(
        valid[None, :], gl.exp2((score - next_max[:, None]) * 1.4426950408889634), 0.0
    )
    denominator = denominator * alpha + gl.sum(probability, 1)
    high = probability.to(gl.bfloat16)
    low = (probability - high.to(gl.float32)).to(gl.bfloat16)
    packed = gl.reshape(gl.permute(gl.join(high, low), (2, 0, 1)), (16, TILE))
    packed_alpha = gl.reshape(gl.permute(gl.join(alpha, alpha), (1, 0)), (16,))
    probability_shared.store(packed)
    alpha_shared.store(packed_alpha)
    value_alpha = alpha_shared.load(gl.SliceLayout(1, MMA))
    p = probability_shared.load(LHS)
    return (p, value_alpha, next_max, denominator)


@gluon.jit
def _update_values(
    p,
    value_alpha,
    accumulators,
    cache_shared,
    early_values,
    KV,
    next_slot,
    refill_channel,
    next_active,
    K0: gl.constexpr,
    copy_shared,
):
    updated = ()
    for chunk in gl.static_range(len(accumulators)):
        if chunk < 2:
            value = early_values[chunk].to(gl.bfloat16)
        else:
            region = cache_shared.slice(chunk * PV_CHUNK, PV_CHUNK, 1)
            value = gl.amd.cdna4.async_copy.load_shared_relaxed(region, RHS).to(
                gl.bfloat16
            )
            gl.barrier()
            _copy_region(
                copy_shared.slice(chunk * PV_CHUNK, PV_CHUNK, 1),
                KV + chunk * PV_CHUNK,
                next_slot,
                refill_channel,
                next_active,
                K0,
                True,
            )
        updated += (
            gl.amd.cdna4.mfma(p, value, accumulators[chunk] * value_alpha[:, None]),
        )
    return updated


@gluon.jit
def _finish_values(O, row, p, value_alpha, accumulators, cache_shared, denominator):
    denominator = gl.convert_layout(denominator, gl.SliceLayout(1, MMA))
    inverse_denom = 1.0 / gl.where(denominator > 0, denominator, 1.0)[:, None]
    head = gl.arange(0, HEADS, gl.SliceLayout(1, MMA))
    channel = gl.arange(0, PV_CHUNK, gl.SliceLayout(0, MMA))
    out = O + (row * HEADS + head[:, None]) * LATENT + channel[None, :]
    for chunk in gl.static_range(LATENT // PV_CHUNK):
        value = gl.amd.cdna4.async_copy.load_shared_relaxed(
            cache_shared.slice(chunk * PV_CHUNK, PV_CHUNK, 1), RHS
        ).to(gl.bfloat16)
        accumulator = gl.amd.cdna4.mfma(
            p, value, accumulators[chunk] * value_alpha[:, None]
        )
        halves = gl.reshape(accumulator, (2, HEADS, PV_CHUNK))
        result = gl.convert_layout(gl.sum(halves, 0), MMA)
        value = (result * inverse_denom).to(gl.bfloat16)
        gl.store(out + chunk * PV_CHUNK, value)


@gluon.jit
def _attention_pipeline(
    Q,
    KV,
    S,
    O,
    scale,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    H: gl.constexpr,
    SELECTED: gl.constexpr,
):
    gl.static_assert(H == HEADS and SELECTED == 2048)
    gl.static_assert(K0 == 576 or K0 == 608)
    row = gl.num_programs(0) - 1 - gl.program_id(0)
    active_tiles, full_tiles, validity_shared = _scan_tiles(S, row, S0)
    if active_tiles == 0:
        empty_layout: gl.constexpr = gl.BlockedLayout([4], [64], [2], [0])
        empty_col = gl.arange(0, HEADS * LATENT, empty_layout)
        zeros = gl.full((HEADS * LATENT,), 0, gl.bfloat16, empty_layout)
        gl.amd.cdna4.buffer_store(zeros, O + row * HEADS * LATENT, empty_col)
        return
    q_latent, q_rope = _load_query(Q, row, Q0, Q1)
    rope = gl.arange(0, 64, gl.SliceLayout(0, COPY))
    q_latent, q_rope = _stage_query(q_latent, q_rope)
    col = gl.arange(0, TILE, gl.SliceLayout(1, COPY))
    maximum = gl.full((HEADS,), -float("inf"), gl.float32, gl.SliceLayout(1, STATS))
    denominator = gl.full((HEADS,), 0, gl.float32, gl.SliceLayout(1, STATS))
    accumulators = ()
    for chunk in gl.static_range(LATENT // PV_CHUNK):
        accumulators += (gl.full((16, PV_CHUNK), 0, gl.float32, MMA),)
    (
        cache_shared,
        copy_shared,
        rope_shared,
        rope_copy,
        probability_shared,
        alpha_shared,
    ) = _allocate_tile_storage(KV.dtype.element_ty)
    block = _first_active(active_tiles)
    slot = gl.amd.cdna4.buffer_load(S + row * S0, block * TILE + col)
    active = slot >= 0
    key_channel = gl.arange(0, QK_CHUNK, gl.SliceLayout(0, COPY))
    for key_chunk in gl.static_range(LATENT // QK_CHUNK):
        _copy_region(
            copy_shared.slice(key_chunk * QK_CHUNK, QK_CHUNK, 1),
            KV + key_chunk * QK_CHUNK,
            slot,
            key_channel,
            active,
            K0,
            True,
        )
    _copy_region(rope_copy, KV + LATENT, slot, rope, active, K0, True)
    gl.amd.cdna4.async_copy.commit_group()
    refill_channel = gl.arange(0, PV_CHUNK, gl.SliceLayout(0, COPY))
    stats_col = gl.arange(0, TILE, gl.SliceLayout(0, STATS))
    while _remaining_tiles(active_tiles) != 0:
        active_tiles = _remaining_tiles(active_tiles)
        next_block = _first_active(active_tiles)
        next_slot = gl.amd.cdna4.buffer_load(S + row * S0, next_block * TILE + col)
        next_active = next_slot >= 0
        score, valid = _tile_scores(
            q_latent,
            q_rope,
            cache_shared,
            rope_shared,
            full_tiles,
            block,
            validity_shared,
            stats_col,
        )
        early_values = ()
        for early in gl.static_range(2):
            early_values += (
                gl.amd.cdna4.async_copy.load_shared_relaxed(
                    cache_shared.slice(early * PV_CHUNK, PV_CHUNK, 1), RHS
                ),
            )
        p, value_alpha, next_max, denominator = _softmax_update(
            score, valid, maximum, denominator, scale, probability_shared, alpha_shared
        )
        _copy_region(
            copy_shared.slice(0, QK_CHUNK, 1),
            KV,
            next_slot,
            key_channel,
            next_active,
            K0,
            True,
        )
        next_accumulators = _update_values(
            p,
            value_alpha,
            accumulators,
            cache_shared,
            early_values,
            KV,
            next_slot,
            refill_channel,
            next_active,
            K0,
            copy_shared,
        )
        _copy_region(rope_copy, KV + LATENT, next_slot, rope, next_active, K0, True)
        gl.amd.cdna4.async_copy.commit_group()
        block = next_block
        accumulators = next_accumulators
        maximum = next_max
    score, valid = _tile_scores(
        q_latent,
        q_rope,
        cache_shared,
        rope_shared,
        full_tiles,
        block,
        validity_shared,
        stats_col,
    )
    p, value_alpha, _, denominator = _softmax_update(
        score, valid, maximum, denominator, scale, probability_shared, alpha_shared
    )
    _finish_values(O, row, p, value_alpha, accumulators, cache_shared, denominator)


def _launch(query, kv_cache, selected_slots, output, softmax_scale):
    m, h, _ = query.shape
    return _attention_pipeline[m,](
        query,
        kv_cache,
        selected_slots,
        output,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        h,
        selected_slots.shape[1],
        num_warps=2,
        waves_per_eu=2,
        enable_fp_fusion=False,
        llvm_fn_attrs=[["amdgpu-agpr-alloc", "0"]],
    )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
    output: torch.Tensor | None = None,
):
    m, h, _ = query.shape
    if output is None:
        output = torch.empty((m, h, 512), dtype=torch.bfloat16, device=query.device)
    _launch(query, kv_cache, selected_slots, output, softmax_scale)
    return output
