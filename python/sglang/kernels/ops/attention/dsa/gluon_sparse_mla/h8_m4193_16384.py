# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m4193_16384.py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import core as tl_core


@gluon.constexpr_function
def _latent_copy_layout(rows):
    assert rows in (32, 64)
    registers = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]]
    if rows == 64:
        registers += [[32, 0]]
    return gl.DistributedLinearLayout(
        registers,
        [[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
        [[1, 0], [2, 0]],
        [],
        [rows, 512],
    )


@gluon.jit
def _merge_tile_flags(a, b):
    return a | b


@gluon.jit
def _wave_priority(level: gl.constexpr):
    gl.inline_asm_elementwise(
        f"s_setprio {level}", "=s", [], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _unique_head_scores(score, layout: gl.constexpr):
    low_columns, high_columns = gl.split(
        score.reshape([16, 16, 2, 2]).permute((0, 1, 3, 2))
    )
    head = gl.arange(
        0, 16, layout=gl.SliceLayout(1, gl.SliceLayout(2, low_columns.type.layout))
    )
    unique = gl.where(head[:, None, None] < 8, low_columns, high_columns)
    unique = unique.reshape([2, 8, 16, 2]).permute((1, 2, 0, 3))
    return gl.convert_layout(unique.reshape([8, 64]), layout, assert_trivial=True)


@gluon.jit
def _broadcast_head_banks(value, layout: gl.constexpr):
    template = gl.full([16], 0.0, gl.float32, layout).reshape([2, 8])
    bank = gl.convert_layout(value, gl.SliceLayout(0, template.type.layout))
    both = gl.broadcast(bank[None, :], template)[0].reshape([16])
    return gl.convert_layout(both, layout, assert_trivial=True)


@gluon.jit
def _sum_head_banks(value):
    width: gl.constexpr = value.shape[1]
    banks = value.reshape([2, 8, width])
    paired = gl.sum(banks, 0)
    both = gl.broadcast(paired[None, :, :], banks)[0]
    return gl.convert_layout(both.reshape([16, width]), value.type.layout)


@gluon.jit
def _first_live_tile(mask):
    return gl.inline_asm_elementwise(
        "s_ff1_i32_b32 $0, $1", "=s,s", [mask], dtype=gl.int32, is_pure=True, pack=1
    )


@gluon.jit
def _scan_selections(
    selection_base, SELECTION_DTYPE: gl.constexpr, SELECTION_CACHE: gl.constexpr
):
    scan_layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    column = gl.arange(0, 2048, layout=scan_layout)
    selected = gl.amd.cdna4.buffer_load(selection_base, column, cache=SELECTION_CACHE)
    compact = gl.maximum(selected, -1).to(SELECTION_DTYPE)
    selection_shared = gl.allocate_shared_memory(
        SELECTION_DTYPE, [2048], gl.SwizzledSharedLayout(1, 1, 1, [0]), compact
    )
    tile_bit = (1 << column // 64).to(gl.uint32)
    tile_flags = gl.where(
        selected >= 0, tile_bit.to(gl.uint64), tile_bit.to(gl.uint64) << 32
    )
    packed_flags = gl.reduce(tile_flags, 0, _merge_tile_flags)
    live_tiles = packed_flags.to(gl.uint32)
    invalid_tiles = (packed_flags >> 32).to(gl.uint32)
    live_tiles, invalid_tiles = gl.inline_asm_elementwise(
        "s_mov_b32 $0, $2\n s_mov_b32 $1, $3",
        "=s,=s,s,s",
        [live_tiles, invalid_tiles],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )
    return (selection_shared, live_tiles, invalid_tiles)


@gluon.jit
def _prepare_row(
    Q,
    S,
    row,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    S0: gl.constexpr,
    SELECTION_DTYPE: gl.constexpr,
    SELECTION_CACHE: gl.constexpr,
    qk_dot_q: gl.constexpr,
):
    q_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 4], [1, 0])
    qh = gl.arange(0, 16, layout=gl.SliceLayout(1, q_layout)) % 8
    qd = gl.arange(0, 512, layout=gl.SliceLayout(0, q_layout))
    qr = gl.arange(0, 64, layout=gl.SliceLayout(0, q_layout))
    latent_head = gl.arange(0, 8, layout=gl.SliceLayout(1, q_layout))
    query = gl.load(Q + row * Q0 + latent_head[:, None] * Q1 + qd[None, :])
    q_shared = gl.allocate_shared_memory(
        gl.bfloat16,
        [32, 128],
        gl.SwizzledSharedLayout(8, 1, 8, [1, 0]),
        query.reshape([8, 2, 2, 128]).permute((2, 1, 0, 3)).reshape([32, 128]),
    )
    query_rotary = gl.load(Q + row * Q0 + qh[:, None] * Q1 + 512 + qr[None, :])
    query_rotary = gl.convert_layout(query_rotary, qk_dot_q)
    selections, remaining, invalid = _scan_selections(
        S + row * S0, SELECTION_DTYPE, SELECTION_CACHE
    )
    resident_queries = ()
    for fragment in gl.static_range(2):
        resident_queries += (q_shared.slice(fragment * 16, 16, 0).load(qk_dot_q),)
    return (resident_queries, query_rotary, selections, remaining, invalid)


@gluon.jit
def _gather_selection(selections, positions):
    slot = selections.gather(positions, 0).to(gl.int32)
    if selections.dtype == gl.uint16:
        slot = gl.where(slot == 65535, -1, slot)
    return slot


@gluon.jit
def _stage_latent(shared, cache, selections, start, K0: gl.constexpr, j):
    slot = _gather_selection(selections, start + j)
    channel = gl.arange(0, 512, layout=gl.SliceLayout(0, j.type.layout.parent))
    offset = slot * K0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        shared, cache, offset[:, None] + channel[None, :]
    )


@gluon.jit
def _load_rotary(cache, selections, start, channel, K0: gl.constexpr):
    j = gl.arange(0, 64, layout=gl.SliceLayout(1, channel.type.layout.parent))
    slot = _gather_selection(selections, start + j)
    offset = slot * K0
    return gl.amd.cdna4.buffer_load(cache, offset[:, None] + 512 + channel[None, :])


@gluon.jit
def _stage_tile(shared, cache, selections, start, rotary_channel, K0: gl.constexpr):
    copy_layout: gl.constexpr = _latent_copy_layout(64)
    j = gl.arange(0, 64, layout=gl.SliceLayout(1, copy_layout))
    _stage_latent(shared, cache, selections, start, K0, j)
    gl.amd.cdna4.async_copy.commit_group()
    return _load_rotary(cache, selections, start, rotary_channel, K0)


@gluon.jit
def _qk_scores(
    kv_shared,
    resident_queries,
    query_rotary,
    rotary,
    scale,
    softmax_layout: gl.constexpr,
):
    mma: gl.constexpr = query_rotary.type.layout.parent
    qk_dot_k: gl.constexpr = gl.DotOperandLayout(
        1, mma, query_rotary.type.layout.k_width
    )
    score_parts = ()
    for half in gl.static_range(2):
        partial = gl.full([16, 64], 0.0, gl.float32, mma)
        keys = ()
        for fragment in gl.static_range(2):
            keys += (
                gl.amd.cdna4.async_copy.load_shared_relaxed(
                    kv_shared.slice(half * 256 + fragment * 128, 128, 1).permute(
                        (1, 0)
                    ),
                    qk_dot_k,
                ).to(gl.bfloat16),
            )
        for fragment in gl.static_range(2):
            partial = gl.amd.cdna4.mfma(
                resident_queries[fragment], keys[fragment], partial
            )
        score_parts += (partial,)
    head = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    selected_parts = gl.where(head[:, None] < 8, score_parts[0], score_parts[1])
    score = _sum_head_banks(selected_parts)
    key_rotary = gl.convert_layout(rotary.permute((1, 0)), qk_dot_k).to(gl.bfloat16)
    score = gl.amd.cdna4.mfma(query_rotary, key_rotary, score)
    return _unique_head_scores(score, softmax_layout) * scale


@gluon.jit
def _mask_scores(score, selections, start, all_active):
    slot_layout: gl.constexpr = gl.SliceLayout(0, score.type.layout)
    if all_active:
        live = gl.full([64], True, gl.int1, slot_layout)
    else:
        j = gl.arange(0, 64, layout=slot_layout)
        live = _gather_selection(selections, start + j) >= 0
        score = gl.where(live[None, :], score, -float("inf"))
    return (score, live)


@gluon.jit
def _maximum_propagating(a, b):
    return gl.maximum(a, b, propagate_nan=tl_core.PropagateNan.ALL)


@gluon.jit
def _online_maximum(score, maximum, maximum_shared, SOFTMAX_PRIORITY: gl.constexpr):
    maximum_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 4], [0, 1])
    local_maximum = gl.reduce(score.reshape([8, 32, 2]), 2, _maximum_propagating)
    prior_maximum = gl.convert_layout(
        maximum, gl.SliceLayout(1, local_maximum.type.layout)
    )
    local_maximum = _maximum_propagating(local_maximum, prior_maximum[:, None])
    quarter_maximum = gl.reduce(
        local_maximum.reshape([8, 4, 4, 2]), 3, _maximum_propagating
    )
    wave_maximum = gl.reduce(quarter_maximum, 2, _maximum_propagating)
    _wave_priority(SOFTMAX_PRIORITY)
    maximum_shared.store(wave_maximum)
    tile_maximum = gl.reduce(
        maximum_shared.load(maximum_layout), 1, _maximum_propagating
    )
    return gl.convert_layout(tile_maximum, maximum.type.layout)


@gluon.jit
def _needs_rescale(alpha):
    changed = gl.inline_asm_elementwise(
        "v_cmp_neq_f32_e64 $0, 1.0, $1",
        "=s,v",
        [alpha],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    vote = gl.gather(changed, gl.full([1], 0, gl.int32, alpha.type.layout), 0)
    return gl.sum(vote, 0) != 0


@gluon.jit
def _softmax_update(
    score,
    live,
    all_active,
    maximum,
    next_maximum,
    probability_shared,
    denominator,
    accumulators,
    first_tile,
):
    mma: gl.constexpr = accumulators[0].type.layout
    alpha = gl.exp2((maximum - next_maximum) * 1.4426950408889634)
    if all_active:
        probability = gl.exp2((score - next_maximum[:, None]) * 1.4426950408889634)
    else:
        probability = gl.where(
            live[None, :],
            gl.exp2((score - next_maximum[:, None]) * 1.4426950408889634),
            0.0,
        )
    tile_denominator = gl.sum(probability.reshape([8, 32, 2]), 2)
    p_high = probability.to(gl.bfloat16)
    p_low = (probability - p_high.to(gl.float32)).to(gl.bfloat16)
    probability_banks = gl.join(p_high, p_low).permute((2, 0, 1)).reshape([16, 64])
    _wave_priority(3)
    probability_shared.store(probability_banks)
    if not first_tile and _needs_rescale(alpha):
        lane_alpha = gl.convert_layout(
            alpha, gl.SliceLayout(1, denominator.type.layout)
        )
        denominator = denominator * lane_alpha[:, None]
        both_alpha = _broadcast_head_banks(alpha, gl.SliceLayout(1, mma))
        rescaled_accumulators = ()
        for segment in gl.static_range(4):
            rescaled_accumulators += (accumulators[segment] * both_alpha[:, None],)
        accumulators = rescaled_accumulators
    return (denominator + tile_denominator, accumulators)


@gluon.jit
def _accumulate_values(
    kv_shared,
    probability_shared,
    accumulators,
    cache,
    selections,
    next_start,
    has_successor,
    half_j,
    K0: gl.constexpr,
    DEFER_VALUE_CAST: gl.constexpr,
):
    mma: gl.constexpr = accumulators[0].type.layout
    pv_dot_p: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    pv_dot_v: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    for reduction_half in gl.static_range(2):
        p_fragment = probability_shared.slice(reduction_half * 32, 32, 1).load(pv_dot_p)
        next_accumulators = ()
        for pair in gl.static_range(2):
            values = ()
            for local in gl.static_range(2):
                value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    kv_shared.slice(reduction_half * 32, 32, 0).slice(
                        (pair * 2 + local) * 128, 128, 1
                    ),
                    pv_dot_v,
                )
                if not DEFER_VALUE_CAST:
                    value = value.to(gl.bfloat16)
                values += (value,)
            if pair == 1 and has_successor:
                gl.barrier()
                _stage_latent(
                    kv_shared.slice(reduction_half * 32, 32, 0),
                    cache,
                    selections,
                    next_start + reduction_half * 32,
                    K0,
                    half_j,
                )
                if reduction_half == 1:
                    gl.amd.cdna4.async_copy.commit_group()
            for local in gl.static_range(2):
                next_accumulators += (
                    gl.amd.cdna4.mfma(
                        p_fragment,
                        values[local].to(gl.bfloat16),
                        accumulators[pair * 2 + local],
                    ),
                )
        accumulators = next_accumulators
    return accumulators


@gluon.jit
def _store_output(O, row, denominator, accumulators):
    mma: gl.constexpr = accumulators[0].type.layout
    oh = gl.arange(0, 16, layout=gl.SliceLayout(1, mma))
    od = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
    denominator = _broadcast_head_banks(gl.sum(denominator, 1), gl.SliceLayout(1, mma))
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    for segment in gl.static_range(4):
        total = _sum_head_banks(accumulators[segment])
        result = total * inverse_denominator[:, None]
        gl.store(
            O + (row * 8 + oh[:, None]) * 512 + segment * 128 + od[None, :],
            result.to(gl.bfloat16),
            oh[:, None] < 8,
        )


@gluon.jit
def _attention_rows(
    Q,
    KV,
    S,
    O,
    scale,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    SELECTION_DTYPE: gl.constexpr,
    SELECTION_CACHE: gl.constexpr,
    QK_PRIORITY: gl.constexpr,
    SOFTMAX_PRIORITY: gl.constexpr,
    DEFER_VALUE_CAST: gl.constexpr,
):
    row = gl.num_programs(0) - 1 - gl.program_id(0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    qk_dot_q: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    pv_dot_v: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    rotary_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [0, 1])
    softmax_layout: gl.constexpr = gl.BlockedLayout([1, 2], [8, 8], [1, 4], [0, 1])
    half_async_layout: gl.constexpr = _latent_copy_layout(32)
    resident_queries, query_rotary, selection_shared, remaining_tiles, invalid_tiles = (
        _prepare_row(Q, S, row, Q0, Q1, S0, SELECTION_DTYPE, SELECTION_CACHE, qk_dot_q)
    )
    kv_shared_layout: gl.constexpr = (
        gl.amd.cdna4.compute_efficient_padded_shared_layout(
            pv_dot_v, [64, 512], KV.dtype.element_ty, is_k_contig=False
        )
    )
    kv_shared = gl.allocate_shared_memory(
        KV.dtype.element_ty, [64, 512], kv_shared_layout
    )
    maximum_shared = gl.allocate_shared_memory(
        gl.float32, [8, 4], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    probability_shared = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    )
    r = gl.arange(0, 64, layout=gl.SliceLayout(0, rotary_layout))
    maximum = gl.full([8], -float("inf"), gl.float32, gl.SliceLayout(1, softmax_layout))
    denominator = gl.sum(
        gl.full([8, 64], 0.0, gl.float32, softmax_layout).reshape([8, 32, 2]), 2
    )
    accumulators = ()
    for segment in gl.static_range(4):
        accumulators += (gl.full([16, 128], 0.0, gl.float32, mma),)
    prefetched_rotary = gl.full([64, 64], 0.0, KV.dtype.element_ty, rotary_layout)
    start = _first_live_tile(remaining_tiles) * 64
    all_active = invalid_tiles & (remaining_tiles & -remaining_tiles) == 0
    if remaining_tiles != 0:
        prefetched_rotary = _stage_tile(kv_shared, KV, selection_shared, start, r, K0)
    first_tile = True
    while remaining_tiles != 0:
        rotary = prefetched_rotary
        gl.amd.cdna4.async_copy.wait_group(0)
        _wave_priority(QK_PRIORITY)
        score = _qk_scores(
            kv_shared, resident_queries, query_rotary, rotary, scale, softmax_layout
        )
        remaining_tiles = remaining_tiles & remaining_tiles - 1
        next_tile = _first_live_tile(remaining_tiles) & 31
        next_start = next_tile * 64
        successor_all_active = (remaining_tiles != 0) & (
            invalid_tiles & 1 << next_tile == 0
        )
        has_successor = remaining_tiles != 0
        half_j = gl.arange(0, 32, layout=gl.SliceLayout(1, half_async_layout))
        score, live = _mask_scores(score, selection_shared, start, all_active)
        next_maximum = _online_maximum(score, maximum, maximum_shared, SOFTMAX_PRIORITY)
        prefetched_rotary = gl.full([64, 64], 0.0, KV.dtype.element_ty, rotary_layout)
        if has_successor:
            prefetched_rotary = _load_rotary(KV, selection_shared, next_start, r, K0)
        denominator, accumulators = _softmax_update(
            score,
            live,
            all_active,
            maximum,
            next_maximum,
            probability_shared,
            denominator,
            accumulators,
            first_tile,
        )
        accumulators = _accumulate_values(
            kv_shared,
            probability_shared,
            accumulators,
            KV,
            selection_shared,
            next_start,
            has_successor,
            half_j,
            K0,
            DEFER_VALUE_CAST,
        )
        first_tile = False
        maximum = next_maximum
        start = next_start
        all_active = successor_all_active
        _wave_priority(0)
    _store_output(O, row, denominator, accumulators)


def _launch_policy(rows, cache_rows):
    return dict(
        SELECTION_DTYPE=gl.int16
        if cache_rows <= 32768
        else gl.uint16
        if cache_rows <= 65535
        else gl.int32,
        SELECTION_CACHE=".cg" if rows >= 12288 else "",
        QK_PRIORITY=1 if rows <= 12288 else 2,
        SOFTMAX_PRIORITY=2 if rows <= 12288 else 1,
        DEFER_VALUE_CAST=rows <= 8192,
    )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
    output: torch.Tensor | None = None,
):
    m, h, d = query.shape
    assert m > 0 and h == 8 and (d == 576) and (query.dtype == torch.bfloat16)
    assert kv_cache.ndim == 2 and kv_cache.shape[1] == 576
    assert kv_cache.dtype == torch.float8_e4m3fn and kv_cache.stride(0) >= 576
    assert selected_slots.shape == (m, 2048) and selected_slots.dtype == torch.int32
    assert query.stride(-1) == kv_cache.stride(-1) == selected_slots.stride(-1) == 1
    assert (
        query.device == kv_cache.device == selected_slots.device and softmax_scale > 0
    )
    assert (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 575 < 2**31
    if output is None:
        output = torch.empty((m, 8, 512), dtype=torch.bfloat16, device=query.device)
    _attention_rows[m,](
        query,
        kv_cache,
        selected_slots,
        output,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        **_launch_policy(m, kv_cache.shape[0]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
