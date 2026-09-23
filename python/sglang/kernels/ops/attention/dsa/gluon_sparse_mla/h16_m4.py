# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m4.py
from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


class _Schedule(NamedTuple):
    profile: int
    block: int
    heads: int
    qk: int
    pv: int
    qk_width: int
    query_vec: int
    waves: int
    merge_cols: int
    merge_heads: int
    merge_warps: int
    padding: int = 0
    column_xor: int = 0


def _schedule(m):
    if m == 1:
        return _Schedule(1, 64, 2, 128, 512, 16, 32, 1, 32, 2, 1)
    if m == 2:
        return _Schedule(2, 64, 4, 256, 512, 16, 32, 1, 16, 4, 1, 192)
    if m <= 4:
        return _Schedule(4, 64, 8, 512, 128, 8, 16, 0, 8, 8, 1)
    if m <= 8:
        return _Schedule(8, 64, 16, 128, 64, 16, 8, 2, 16, 16, 4, 96)
    return _Schedule(16, 128, 16, 256, 64, 16, 8, 0, 32, 8, 2, 96, 128)


@gluon.jit
def _probability_operands(p, DOT: gl.constexpr, FP16: gl.constexpr):
    DTYPE: gl.constexpr = gl.float16 if FP16 else gl.bfloat16
    RESIDUAL_SCALE: gl.constexpr = 16384.0 if FP16 else 1.0
    high = p.to(DTYPE)
    low = ((p - high.to(gl.float32)) * RESIDUAL_SCALE).to(DTYPE)
    if FP16:
        packed = (
            high.to(gl.uint16, bitcast=True).to(gl.uint32)
            | low.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
        )
        packed = gl.convert_layout(packed, DOT)
        high_operand = packed.to(gl.uint16).to(DTYPE, bitcast=True)
        low_operand = (packed >> 16).to(gl.uint16).to(DTYPE, bitcast=True)
    else:
        high_operand = gl.convert_layout(high, DOT)
        low_operand = gl.convert_layout(low, DOT)
    return (high_operand, low_operand)


@gluon.jit
def _component_probability(p, DOT: gl.constexpr):
    high = p.to(gl.bfloat16)
    low = (p - high.to(gl.float32)).to(gl.bfloat16)
    paired = gl.join(low, high).permute((2, 0, 1)).reshape((2 * p.shape[0], p.shape[1]))
    return gl.convert_layout(paired, DOT)


@gluon.jit
def _first_band(a, b):
    return a


@gluon.jit
def _last_band(a, b):
    return b


@gluon.jit
def _component_value(
    probability, value, ROWS: gl.constexpr, COLS: gl.constexpr, MMA: gl.constexpr
):
    components = gl.amd.cdna4.mfma(
        probability, value, gl.zeros((2 * ROWS, COLS), gl.float32, MMA)
    )
    bands = components.reshape((2, ROWS, COLS))
    low = gl.reduce(bands, 0, _first_band)
    high = gl.reduce(bands, 0, _last_band)
    numerator = low + high
    return gl.convert_layout(numerator, MMA)


@gluon.jit
def _separate_value(
    low, high, value, ROWS: gl.constexpr, COLS: gl.constexpr, MMA: gl.constexpr
):
    low_acc = gl.amd.cdna4.mfma(low, value, gl.zeros((ROWS, COLS), gl.float32, MMA))
    high_acc = gl.amd.cdna4.mfma(high, value, gl.zeros((ROWS, COLS), gl.float32, MMA))
    return low_acc * (1.0 / 16384.0) + high_acc


@gluon.jit
def _store_statistics(maximum, denominator, Stats, record):
    packed = (
        maximum.to(gl.uint32, bitcast=True).to(gl.uint64)
        | denominator.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
    )
    gl.amd.cdna4.buffer_store(
        packed, Stats.to(gl.pointer_type(gl.uint64)), record, None
    )


@gluon.jit
def _store_numerator(
    numerator,
    denominator,
    Partials,
    row,
    head_group,
    split,
    hm,
    vm,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    C: gl.constexpr,
):
    gl.static_assert(C.column_xor % 4 == 0)
    vm_group = vm // 4 ^ split * (C.column_xor // 4) % 128
    if C.profile == 1:
        split_record = (row * gl.cdiv(HEADS, C.heads) + head_group) * SPLITS + split
        Partials = Partials + split_record * (C.heads * 512 + C.padding)
        offsets = (
            vm_group[None, :] * C.heads * 4
            + (hm[:, None] - head_group * C.heads) * 4
            + vm[None, :] % 4
        )
    elif C.profile <= 4:
        offsets = (
            (
                ((row * gl.cdiv(HEADS, C.heads) + head_group) * SPLITS + split) * 128
                + vm_group[None, :]
            )
            * C.heads
            * 4
            + (hm[:, None] - head_group * C.heads) * 4
            + vm[None, :] % 4
        )
    else:
        offsets = (
            ((row * SPLITS + split) * 128 + vm_group[None, :]) * HEADS * 4
            + hm[:, None] * 4
            + vm[None, :] % 4
        )
    if C.padding and C.profile != 1:
        if C.profile <= 4:
            split_record = (row * gl.cdiv(HEADS, C.heads) + head_group) * SPLITS + split
        else:
            split_record = row * SPLITS + split
        offsets = offsets + split_record * C.padding
    active_heads = denominator != 0.0
    gl.amd.cdna4.buffer_store(
        numerator,
        Partials,
        offsets,
        active_heads[:, None],
        cache=".cs" if C.profile == 2 or C.profile > 4 else "",
    )


@gluon.jit
def _stage_query(
    Q,
    row,
    head_group,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    qk_a: gl.constexpr,
    C: gl.constexpr,
):
    QUERY_ROWS: gl.constexpr = C.heads
    if C.profile <= 4:
        q_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 4], [1, 0])
    else:
        q_load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    qlh = head_group * C.heads + gl.arange(
        0, QUERY_ROWS, layout=gl.SliceLayout(1, q_load_layout)
    )
    qld = gl.arange(0, 512, layout=gl.SliceLayout(0, q_load_layout))
    if C.profile > 4:
        query_base = Q + row * Q0 + head_group * C.heads * Q1
        q = gl.amd.cdna4.buffer_load(
            query_base,
            (qlh[:, None] - head_group * C.heads) * Q1 + qld[None, :],
            None,
            None,
        )
    else:
        q = gl.amd.cdna4.buffer_load(
            Q, row * Q0 + qlh[:, None] * Q1 + qld[None, :], None, None
        )
    q_shared = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        (QUERY_ROWS, 512),
        gl.SwizzledSharedLayout(C.query_vec, 1, 16, [1, 0]),
        q,
    )
    qh = head_group * C.heads + gl.arange(0, QUERY_ROWS, layout=gl.SliceLayout(1, qk_a))
    qdr = gl.arange(0, 64, layout=gl.SliceLayout(0, qk_a))
    if C.profile > 4:
        rotary_base = Q + row * Q0 + head_group * C.heads * Q1 + 512
        qr = gl.amd.cdna4.buffer_load(
            rotary_base,
            (qh[:, None] - head_group * C.heads) * Q1 + qdr[None, :],
            None,
            None,
        )
    else:
        qr = gl.amd.cdna4.buffer_load(
            Q, row * Q0 + qh[:, None] * Q1 + 512 + qdr[None, :], None, None
        )
    return (q_shared, qr)


@gluon.jit
def _value_products(
    p,
    kv_shared,
    denominator,
    Partials,
    row,
    head_group,
    split,
    hm,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    mma_layout: gl.constexpr,
    C: gl.constexpr,
):
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    pv_dtype: gl.constexpr = gl.float16 if C.profile == 8 else gl.bfloat16
    if C.pv == 512:
        value = kv_shared.load(dot_b).to(pv_dtype)
    elif C.profile == 4 or C.profile == 16:
        value_raw = kv_shared.slice(0, C.pv, dim=1).load(dot_b)
        if C.profile != 4:
            value = value_raw.to(pv_dtype)
    if C.profile <= 4:
        probability = _component_probability(p, dot_a)
    else:
        high, low = _probability_operands(p, dot_a, C.profile == 8)
    for vi in gl.static_range(512 // C.pv):
        vm = vi * C.pv + gl.arange(0, C.pv, layout=gl.SliceLayout(0, mma_layout))
        if C.profile == 4 or C.profile == 16:
            if vi + 1 < 512 // C.pv:
                next_raw = kv_shared.slice((vi + 1) * C.pv, C.pv, dim=1).load(dot_b)
                if C.profile != 4:
                    next_value = next_raw.to(pv_dtype)
        elif C.pv != 512:
            value = kv_shared.slice(vi * C.pv, C.pv, dim=1).load(dot_b).to(pv_dtype)
        if C.profile == 4:
            value = value_raw.to(pv_dtype)
        if C.profile <= 4:
            numerator = _component_value(probability, value, C.heads, C.pv, mma_layout)
        elif C.profile == 8:
            numerator = _separate_value(low, high, value, C.heads, C.pv, mma_layout)
        else:
            numerator = gl.amd.cdna4.mfma(
                low, value, gl.zeros((C.heads, C.pv), gl.float32, mma_layout)
            )
            numerator = gl.amd.cdna4.mfma(high, value, numerator)
        _store_numerator(
            numerator,
            denominator,
            Partials,
            row,
            head_group,
            split,
            hm,
            vm,
            HEADS,
            SPLITS,
            C,
        )
        if (C.profile == 4 or C.profile == 16) and vi + 1 < 512 // C.pv:
            if C.profile == 4:
                value_raw = next_raw
            else:
                value = next_value


@gluon.jit
def _attention_partials(
    Q,
    KV,
    Slots,
    Partials,
    Stats,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    HEADS: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    scale,
    C: gl.constexpr,
):
    QUERY_ROWS: gl.constexpr = C.heads
    if C.profile <= 4:
        gl.static_assert(C.heads == 2 or C.heads == 4 or C.heads == 8)
    gl.static_assert(SELECTED % C.block == 0 and HEADS % C.heads == 0)
    row = 0 if C.profile == 1 else gl.program_id(0)
    head_group = 0 if C.profile > 4 and HEADS == C.heads else gl.program_id(1)
    split = gl.program_id(2)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, C.qk_width)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, C.qk_width)
    d = gl.arange(0, 512, layout=gl.SliceLayout(0, load_layout))
    hm = head_group * C.heads + gl.arange(
        0, QUERY_ROWS, layout=gl.SliceLayout(1, mma_layout)
    )
    if C.profile <= 4:
        record = (
            ((row * gl.cdiv(HEADS, C.heads) + head_group) * SPLITS + split) * C.heads
            + hm
            - head_group * C.heads
        )
    else:
        record = (row * SPLITS + split) * HEADS + hm
    jl = split * C.block + gl.arange(0, C.block, layout=gl.SliceLayout(1, load_layout))
    raw_slot = gl.amd.cdna4.buffer_load(Slots, row * S0 + jl)
    active = raw_slot >= 0
    slot = raw_slot.to(gl.int32)
    q_shared, qr = _stage_query(Q, row, head_group, Q0, Q1, qk_a, C)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
        16, 1, 8 if C.profile == 8 else 16, [1, 0]
    )
    raw_kv = gl.amd.cdna4.buffer_load(
        KV, slot[:, None] * K0 + d[None, :], active[:, None], None
    )
    kv_shared = gl.allocate_shared_memory(
        KV.dtype.element_ty, (C.block, 512), shared_layout, raw_kv
    )
    kr_d = gl.arange(0, 64, layout=gl.SliceLayout(1, qk_b))
    jr = split * C.block + gl.arange(0, C.block, layout=gl.SliceLayout(0, qk_b))
    kr_raw_slot = gl.amd.cdna4.buffer_load(Slots, row * S0 + jr)
    kr_slot = kr_raw_slot.to(gl.int32)
    kr_active = kr_raw_slot >= 0
    kr = gl.amd.cdna4.buffer_load(
        KV, kr_slot[None, :] * K0 + 512 + kr_d[:, None], kr_active[None, :], None
    ).to(gl.bfloat16)
    if C.qk == 512:
        q = q_shared.load(qk_a)
        key = kv_shared.permute((1, 0)).load(qk_b).to(gl.bfloat16)
        score = gl.amd.cdna4.mfma(
            q, key, gl.zeros((QUERY_ROWS, C.block), gl.float32, mma_layout)
        )
    else:
        score = gl.zeros((QUERY_ROWS, C.block), gl.float32, mma_layout)
        for ki in gl.static_range(512 // C.qk):
            q = q_shared.slice(ki * C.qk, C.qk, dim=1).load(qk_a)
            key = (
                kv_shared.slice(ki * C.qk, C.qk, dim=1)
                .permute((1, 0))
                .load(qk_b)
                .to(gl.bfloat16)
            )
            score = gl.amd.cdna4.mfma(q, key, score)
    score = gl.amd.cdna4.mfma(qr, kr, score)
    js = split * C.block + gl.arange(0, C.block, layout=gl.SliceLayout(0, mma_layout))
    active_mma = gl.amd.cdna4.buffer_load(Slots, row * S0 + js) >= 0
    score = gl.where(active_mma[None, :], score * scale, -float("inf"))
    maximum = gl.max(score, 1)
    p = gl.where(
        active_mma[None, :],
        gl.exp2((score - maximum[:, None]) * 1.4426950408889634),
        0.0,
    )
    denominator = gl.sum(p, 1)
    if C.profile == 4:
        _store_statistics(maximum, denominator, Stats, record)
    _value_products(
        p,
        kv_shared,
        denominator,
        Partials,
        row,
        head_group,
        split,
        hm,
        HEADS,
        SPLITS,
        mma_layout,
        C,
    )
    if C.profile != 4:
        if C.profile == 2:
            gl.static_assert(C.padding >= 2 * C.heads)
            inline_record = (
                row * gl.cdiv(HEADS, C.heads) + head_group
            ) * SPLITS + split
            record = (
                (inline_record * (C.heads * 512 + C.padding) + C.heads * 512) // 2
                + hm
                - head_group * C.heads
            )
        _store_statistics(maximum, denominator, Stats, record)


@gluon.jit
def _merge_attention(
    Partials,
    Stats,
    Out,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    PAD_SPLITS: gl.constexpr,
    C: gl.constexpr,
):
    GROUP_HEAD_BLOCK: gl.constexpr = C.heads if C.profile <= 4 else 0
    gl.static_assert(HEADS % C.merge_heads == 0)
    row = 0 if C.profile == 1 else gl.program_id(0)
    head_group = gl.program_id(1)
    col_block = gl.program_id(2)
    split_lanes: gl.constexpr = 2 if C.profile == 16 else 4
    layout: gl.constexpr = gl.BlockedLayout(
        [2 if C.profile > 4 else 1, 1, 4],
        [split_lanes, C.merge_heads, 64 // (split_lanes * C.merge_heads)],
        [1, 1, C.merge_warps],
        [1, 2, 0],
    )
    stats_layout: gl.constexpr = gl.SliceLayout(2, layout)
    s = gl.arange(0, PAD_SPLITS, layout=gl.SliceLayout(1, stats_layout))
    h = head_group * C.merge_heads + gl.arange(
        0, C.merge_heads, layout=gl.SliceLayout(0, stats_layout)
    )
    v = col_block * C.merge_cols + gl.arange(
        0, C.merge_cols, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    if GROUP_HEAD_BLOCK:
        record = (
            (row * gl.cdiv(HEADS, GROUP_HEAD_BLOCK) + h[None, :] // GROUP_HEAD_BLOCK)
            * SPLITS
            + s[:, None]
        ) * GROUP_HEAD_BLOCK + h[None, :] % GROUP_HEAD_BLOCK
    else:
        record = (row * SPLITS + s[:, None]) * HEADS + h[None, :]
    if C.profile == 2:
        gl.static_assert(GROUP_HEAD_BLOCK and GROUP_HEAD_BLOCK % C.merge_heads == 0)
        inline_base = (
            row * gl.cdiv(HEADS, GROUP_HEAD_BLOCK)
            + head_group * C.merge_heads // GROUP_HEAD_BLOCK
        ) * SPLITS
        stats_pitch: gl.constexpr = GROUP_HEAD_BLOCK * 512 + C.padding
        inline_head = (
            h - head_group * C.merge_heads // GROUP_HEAD_BLOCK * GROUP_HEAD_BLOCK
        )
        Stats = Stats + inline_base * stats_pitch + GROUP_HEAD_BLOCK * 512
        record = s[:, None] * (stats_pitch // 2) + inline_head[None, :]
    stats_mask = s[:, None] < SPLITS
    packed = gl.amd.cdna4.buffer_load(
        Stats.to(gl.pointer_type(gl.uint64)), record, stats_mask, 4286578688
    )
    maximum = packed.to(gl.uint32).to(gl.float32, bitcast=True)
    denominator = (packed >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
    active = (denominator != 0.0).to(gl.float32)
    global_max = gl.max(maximum, 0)
    if C.profile > 4:
        safe_max = gl.where(global_max != -float("inf"), global_max, 0.0)
    else:
        safe_max = gl.where(gl.sum(active, 0) > 0.0, global_max, 0.0)
    if C.profile == 2:
        weight = gl.where(
            active > 0.0,
            gl.exp2((maximum - safe_max[None, :]) * 1.4426950408889634),
            0.0,
        )
    else:
        weight = gl.where(active > 0.0, gl.exp(maximum - safe_max[None, :]), 0.0)
    numerator_row = row
    if C.profile == 8:
        gl.static_assert(not GROUP_HEAD_BLOCK)
        Partials = Partials + row * SPLITS * (HEADS * 512 + C.padding)
        numerator_row = 0
    gl.static_assert(C.column_xor % 4 == 0)
    mapped_group = v[None, None, :] // 4 ^ s[:, None, None] * (C.column_xor // 4) % 128
    if GROUP_HEAD_BLOCK:
        offsets = (
            (
                (
                    (
                        numerator_row * gl.cdiv(HEADS, GROUP_HEAD_BLOCK)
                        + h[None, :, None] // GROUP_HEAD_BLOCK
                    )
                    * SPLITS
                    + s[:, None, None]
                )
                * 128
                + mapped_group
            )
            * GROUP_HEAD_BLOCK
            * 4
            + h[None, :, None] % GROUP_HEAD_BLOCK * 4
            + v[None, None, :] % 4
        )
    else:
        offsets = (
            ((numerator_row * SPLITS + s[:, None, None]) * 128 + mapped_group)
            * HEADS
            * 4
            + h[None, :, None] * 4
            + v[None, None, :] % 4
        )
    if C.padding:
        if GROUP_HEAD_BLOCK:
            split_record = (
                numerator_row * gl.cdiv(HEADS, GROUP_HEAD_BLOCK)
                + h[None, :, None] // GROUP_HEAD_BLOCK
            ) * SPLITS + s[:, None, None]
        else:
            split_record = numerator_row * SPLITS + s[:, None, None]
        offsets = offsets + split_record * C.padding
    numerator_mask = (s[:, None, None] < SPLITS) & (active[:, :, None] > 0.0)
    numerator = gl.amd.cdna4.buffer_load(
        Partials,
        offsets,
        numerator_mask,
        None if C.profile == 2 or C.profile > 4 else 0.0,
        cache=".cg" if C.profile == 2 or C.profile > 4 else "",
    )
    total = gl.sum(denominator * weight, 0)
    inverse_total = 1.0 / gl.where(total > 0.0, total, 1.0)
    inverse_total = gl.convert_layout(
        inverse_total, gl.SliceLayout(1, gl.SliceLayout(0, layout))
    )
    if C.profile > 4:
        normalized_weight = (
            weight
            * gl.convert_layout(inverse_total, gl.SliceLayout(0, stats_layout))[None, :]
        )
        result = gl.sum(numerator * normalized_weight[:, :, None], 0)
    else:
        result = gl.sum(numerator * weight[:, :, None], 0) * inverse_total[:, None]
    ho = head_group * C.merge_heads + gl.arange(
        0, C.merge_heads, layout=gl.SliceLayout(1, gl.SliceLayout(0, layout))
    )
    vo = col_block * C.merge_cols + gl.arange(
        0, C.merge_cols, layout=gl.SliceLayout(0, gl.SliceLayout(0, layout))
    )
    gl.amd.cdna4.buffer_store(
        result.to(gl.bfloat16),
        Out,
        (row * HEADS + ho[:, None]) * 512 + vo[None, :],
        None,
    )


def _allocate_workspace(query, selected, c):
    m, heads, _ = query.shape
    splits = triton.cdiv(selected, c.block)
    output = torch.empty((m, heads, 512), dtype=torch.bfloat16, device=query.device)
    if c.profile <= 4:
        records = (m, triton.cdiv(heads, c.heads), splits)
        record_heads = c.heads
    else:
        records = (m, splits)
        record_heads = heads
    tail = (record_heads * 512 + c.padding,) if c.padding else (128, record_heads, 4)
    partials = torch.empty(records + tail, dtype=torch.float32, device=query.device)
    stats = (
        partials
        if c.profile == 2
        else torch.empty(
            records + (record_heads, 2), dtype=torch.float32, device=query.device
        )
    )
    return (output, partials, stats)


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, heads, _ = query.shape
    selected = selected_slots.shape[1]
    c = _schedule(m)
    splits = triton.cdiv(selected, c.block)
    output, partials, stats = _allocate_workspace(query, selected, c)
    _attention_partials[m, triton.cdiv(heads, c.heads), splits](
        query,
        kv_cache,
        selected_slots,
        partials,
        stats,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        heads,
        selected,
        splits,
        softmax_scale,
        c,
        num_warps=4,
        enable_fp_fusion=c.profile == 8,
        waves_per_eu=c.waves,
    )
    _merge_attention[m, triton.cdiv(heads, c.merge_heads), 512 // c.merge_cols](
        partials,
        stats,
        output,
        heads,
        splits,
        triton.next_power_of_2(splits),
        c,
        num_warps=c.merge_warps,
        enable_fp_fusion=c.profile > 4,
    )
    return output
