# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m128.py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
)
_SOFTMAX = gl.constexpr(gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0]))
_SOFTMAX_WIDE = gl.constexpr(gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0]))
_PV_OPERAND = gl.constexpr(gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0]))
_QK_OPERAND = gl.constexpr(gl.BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0]))
_QUERY_SHARED = gl.constexpr(gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
_PROBABILITY_SHARED = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 8, [1, 0]))
_LOG2_E = gl.constexpr(1.4426950408889634)
_NARROW_COPY = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[64, 128],
    )
)
_WIDE_COPY = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [64, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[128, 128],
    )
)


@gluon.jit
def _load_query(
    Q,
    row,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    HEADS: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    if BLOCK_N == 128:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 2], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    h = gl.arange(0, 16, gl.SliceLayout(1, layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, layout))
    r = gl.arange(0, 64, gl.SliceLayout(0, layout))
    if BLOCK_N == 128:
        q = gl.load(
            Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < HEADS, 0.0
        )
        qr = gl.load(
            Q + row * Q_ROW + h[:, None] * Q_HEAD + 512 + r[None, :],
            h[:, None] < HEADS,
            0.0,
        )
    else:
        q = gl.amd.cdna4.buffer_load(
            Q + row * Q_ROW, h[:, None] * Q_HEAD + d[None, :], h[:, None] < HEADS, 0.0
        )
        qr = gl.amd.cdna4.buffer_load(
            Q + row * Q_ROW,
            h[:, None] * Q_HEAD + 512 + r[None, :],
            h[:, None] < HEADS,
            0.0,
        )
    return (q, qr)


@gluon.jit
def _load_slots(
    Slots,
    row,
    part,
    S_ROW: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TILES: gl.constexpr,
    KV_ROW: gl.constexpr,
):
    gl.static_assert(TILES == 8)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    tile = gl.arange(0, TILES, gl.SliceLayout(1, load_layout))
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, load_layout))
    offsets = (tile[:, None] * SPLITS + part) * BLOCK_N + n[None, :]
    slots = gl.load(Slots + row * S_ROW + offsets)
    if BLOCK_N == 64:
        ballots = gl.inline_asm_elementwise(
            "v_cmp_ge_i32_e64 $0, $1, 0;",
            constraints="=s,v",
            args=[slots],
            dtype=gl.uint64,
            is_pure=True,
            pack=1,
        )
        first = gl.full((TILES, 1), 0, gl.int32, load_layout)
        tile_masks = gl.gather(ballots, first, 1).reshape((TILES,))
        scalar_layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
        tile_masks = gl.convert_layout(tile_masks, scalar_layout)
        group_masks = ()
        for index in gl.static_range(TILES):
            group_masks += (
                gl.sum(
                    gl.where(
                        gl.arange(0, TILES, scalar_layout) == index, tile_masks, 0
                    ),
                    0,
                ),
            )
    else:
        active = gl.max(slots, 1) >= 0
        bitmap = gl.sum(gl.where(active, 1 << tile, 0), 0)
        group_masks = ()
    if BLOCK_N == 64:
        slots = gl.where(slots >= 0, slots * KV_ROW, -2147483648)
    if BLOCK_N == 128:
        unpack_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 8], [0, 64], [1, 0], [2, 0], [4, 0]],
            lane_bases=[[0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 4]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[8, 128],
        )
        copy_layout: gl.constexpr = _WIDE_COPY
    else:
        unpack_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 8], [1, 0], [2, 0], [4, 0]],
            lane_bases=[[0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 4]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[8, 64],
        )
        copy_layout: gl.constexpr = _NARROW_COPY
    slots = gl.convert_layout(slots, unpack_layout).trans()
    s0246, s1357 = gl.split(slots.reshape((BLOCK_N, 4, 2)))
    s04, s26 = gl.split(s0246.reshape((BLOCK_N, 2, 2)))
    s15, s37 = gl.split(s1357.reshape((BLOCK_N, 2, 2)))
    s0, s4 = gl.split(s04)
    s2, s6 = gl.split(s26)
    s1, s5 = gl.split(s15)
    s3, s7 = gl.split(s37)
    unpacked = (s0, s1, s2, s3, s4, s5, s6, s7)
    group_slots, group_active = ((), ())
    for index in gl.static_range(TILES):
        group_slots += (
            gl.convert_layout(
                unpacked[index], gl.SliceLayout(1, copy_layout), assert_trivial=True
            ),
        )
        if BLOCK_N == 64:
            group_active += (group_masks[index] != 0,)
        else:
            group_active += (bitmap & 1 << index != 0,)
    return (group_slots, group_active, group_masks)


@gluon.jit
def _read_query_panels(query_smem, rotary_query_smem):
    lhs: gl.constexpr = gl.DotOperandLayout(0, _MMA, 16)
    panels = ()
    for fragment in gl.static_range(4):
        panels += (query_smem.slice(fragment * 128, 128, dim=1).load(lhs),)
    panels += (rotary_query_smem.load(lhs),)
    query_smem._keep_alive()
    rotary_query_smem._keep_alive()
    return panels


@gluon.jit
def _prefetch_panel(
    dest,
    KV,
    slots,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    CHANNEL: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    if CHANNEL == 512:
        WIDTH: gl.constexpr = 64
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    else:
        WIDTH: gl.constexpr = 128
        if BLOCK_N == 128:
            layout: gl.constexpr = _WIDE_COPY
        else:
            layout: gl.constexpr = _NARROW_COPY
    slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
    columns = gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    rows = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    flat = dest._reinterpret(
        KV.dtype.element_ty, [BLOCK_N, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    physical_columns = columns[None, :] ^ rows[:, None] % (WIDTH // 16) * 16
    columns_offset = CHANNEL + gl.max_contiguous(
        gl.multiple_of(physical_columns, [1, 16]), [1, 16]
    )
    gl.static_assert(KV_ROWS <= 16777216 and KV_ROW > 0 and (KV_ROW < 16777216))
    gl.static_assert(KV_ROW % 16 == 0)
    gl.static_assert((KV_ROWS - 1) * KV_ROW + 575 < 2147483647)
    if BLOCK_N == 64:
        offsets = slots[:, None].to(gl.uint32) + columns_offset
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            flat, KV, offsets, cache_modifier=".cv"
        )
    else:
        offsets = gl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3;",
            constraints="=v,v,s,v",
            args=[slots[:, None].to(gl.uint32), KV_ROW, columns_offset],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            flat, KV, offsets, slots[:, None] >= 0, cache_modifier=""
        )


@gluon.jit
def _qk_panel(query, key_smem, score):
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, _MMA, 16)
    key = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_smem.permute((1, 0)), _QK_OPERAND
    ).to(gl.bfloat16)
    return gl.amd.cdna4.mfma(query, gl.convert_layout(key, rhs_layout), score)


@gluon.jit
def _online_softmax(
    score,
    valid,
    maximum,
    denominator,
    rotary_smem,
    scale,
    BLOCK_N: gl.constexpr,
    INITIAL: gl.constexpr,
):
    mma_layout: gl.constexpr = score.type.layout
    softmax_layout: gl.constexpr = valid.type.layout.parent
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    score_smem = rotary_smem._reinterpret(
        gl.float32, [16, BLOCK_N], gl.SwizzledSharedLayout(4, 1, 16, [1, 0])
    )
    score_smem.store(score)
    score = gl.amd.cdna4.async_copy.load_shared_relaxed(score_smem, softmax_layout)
    score_smem._keep_alive()
    score = gl.where(valid[None, :], score * scale, -float("inf"))
    next_maximum = gl.maximum(maximum, gl.max(score, 1))
    safe_maximum = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
    if not INITIAL:
        alpha = gl.exp2((maximum - safe_maximum) * _LOG2_E)
        alpha_smem = gl.allocate_shared_memory(
            gl.float32, [16], gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        alpha_smem.store(alpha)
    if BLOCK_N == 128:
        probability = gl.exp2((score - safe_maximum[:, None]) * _LOG2_E)
    else:
        probability = gl.where(
            valid[None, :], gl.exp2((score - safe_maximum[:, None]) * _LOG2_E), 0.0
        )
    if INITIAL:
        denominator = gl.sum(probability, 1)
    else:
        denominator = denominator * alpha + gl.sum(probability, 1)
    probability_smem = rotary_smem._reinterpret(
        gl.bfloat16, [32, BLOCK_N], _PROBABILITY_SHARED
    ).slice(0, 16, dim=0)
    probability_smem.store(probability.to(gl.bfloat16))
    p = gl.amd.cdna4.async_copy.load_shared_relaxed(probability_smem, lhs_layout)
    if INITIAL:
        alpha_mma = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma_layout))
    else:
        alpha_mma = gl.amd.cdna4.async_copy.load_shared_relaxed(
            alpha_smem, gl.SliceLayout(1, mma_layout)
        )
        alpha_smem._keep_alive()
    probability_smem._keep_alive()
    return (p, alpha_mma, next_maximum, denominator)


@gluon.jit
def _store_partial(
    value,
    Partial,
    row,
    part,
    head,
    PANEL: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [0, 1])
    value = value.reshape((16, 2, 16, 4)).permute((0, 2, 1, 3)).reshape((16, 128))
    value = gl.convert_layout(value, layout, assert_trivial=True)
    head = gl.convert_layout(head, layout)
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    base = Partial + ((row * 4 + PANEL) * SPLITS + part) * HEADS * 128
    offsets = d[None, :] // 8 * HEADS * 8 + head * 8 + d[None, :] % 8
    gl.amd.cdna4.buffer_store(
        value.to(gl.bfloat16), base, offsets, head < HEADS, cache=".wt"
    )


@gluon.jit
def _initial_state(SOFTMAX: gl.constexpr = _SOFTMAX):
    numerators = ()
    for fragment in gl.static_range(4):
        numerators += (gl.zeros((16, 128), gl.float32, _MMA),)
    maximum = gl.full((16,), -float("inf"), gl.float32, gl.SliceLayout(1, SOFTMAX))
    denominator = gl.zeros((16,), gl.float32, gl.SliceLayout(1, SOFTMAX))
    return (numerators, maximum, denominator)


@gluon.jit
def _load_key0(KV, slots, KV_ROW: gl.constexpr, BLOCK_N: gl.constexpr):
    if BLOCK_N == 128:
        layout: gl.constexpr = _WIDE_COPY
    else:
        layout: gl.constexpr = _NARROW_COPY
    slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
    columns = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if BLOCK_N == 64:
        offsets = slots[:, None].to(gl.uint32) + columns[None, :]
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
        return gl.amd.cdna4.buffer_load(KV, offsets, cache=".cv")
    else:
        offsets = gl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3;",
            constraints="=v,v,s,v",
            args=[slots[:, None].to(gl.uint32), KV_ROW, columns[None, :]],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
        return gl.amd.cdna4.buffer_load(KV, offsets, slots[:, None] >= 0, 0.0, cache="")


@gluon.jit
def _pipeline_attention(
    KV,
    q,
    qr,
    Slots,
    row,
    part,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    S_ROW: gl.constexpr,
    SPLITS: gl.constexpr,
    scale,
    BLOCK_N: gl.constexpr,
    TILES: gl.constexpr,
):
    softmax_layout: gl.constexpr = _SOFTMAX_WIDE if BLOCK_N == 128 else _SOFTMAX
    group_slots, group_active, group_masks = _load_slots(
        Slots, row, part, S_ROW, SPLITS, BLOCK_N, TILES, KV_ROW
    )
    rhs: gl.constexpr = gl.DotOperandLayout(1, _MMA, 8)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    rotary_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    key0 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    key1 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    if group_active[0]:
        _prefetch_panel(key0, KV, group_slots[0], KV_ROW, KV_ROWS, 0, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
        _prefetch_panel(key1, KV, group_slots[0], KV_ROW, KV_ROWS, 128, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
    query_smem = gl.allocate_shared_memory(gl.bfloat16, [16, 512], _QUERY_SHARED, q)
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], _QUERY_SHARED, qr
    )
    query_panels = _read_query_panels(query_smem, rotary_query_smem)
    numerators, maximum, denominator = _initial_state(softmax_layout)
    key2 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    key3 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_layout
    )
    key_panels = (key0, key1, key2, key3, rotary)
    if group_active[0]:
        for panel in gl.static_range(2, 5):
            _prefetch_panel(
                key_panels[panel],
                KV,
                group_slots[0],
                KV_ROW,
                KV_ROWS,
                panel * 128,
                BLOCK_N,
            )
            gl.amd.cdna4.async_copy.commit_group()
    for step in gl.static_range(TILES):
        slots, active = (group_slots[step], group_active[step])
        if step + 1 < TILES:
            next_slots, next_active = (group_slots[step + 1], group_active[step + 1])
        if active:
            score = gl.zeros((16, BLOCK_N), gl.float32, _MMA)
            for panel in gl.static_range(5):
                gl.amd.cdna4.async_copy.wait_group(4 - panel)
                if BLOCK_N == 128 and panel == 4 and (step + 1 < TILES):
                    next_key0 = _load_key0(KV, next_slots, KV_ROW, BLOCK_N)
                score = _qk_panel(query_panels[panel], key_panels[panel], score)
            if BLOCK_N == 64 and step + 1 < TILES:
                next_key0 = _load_key0(KV, next_slots, KV_ROW, BLOCK_N)
            if BLOCK_N == 64:
                token = gl.arange(0, BLOCK_N, gl.SliceLayout(0, softmax_layout))
                valid = group_masks[step] >> token.to(gl.uint64) & 1 != 0
            else:
                valid = gl.convert_layout(slots, gl.SliceLayout(0, softmax_layout)) >= 0
            p, alpha, maximum, denominator = _online_softmax(
                score, valid, maximum, denominator, rotary, scale, BLOCK_N, step == 0
            )
            next_numerators = ()
            for panel in gl.static_range(4):
                value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    key_panels[panel], _PV_OPERAND
                ).to(gl.bfloat16)
                if step + 1 < TILES:
                    if next_active:
                        gl.barrier()
                        if panel == 0:
                            key0.store(next_key0)
                            gl.barrier()
                        else:
                            _prefetch_panel(
                                key_panels[panel],
                                KV,
                                next_slots,
                                KV_ROW,
                                KV_ROWS,
                                panel * 128,
                                BLOCK_N,
                            )
                            gl.amd.cdna4.async_copy.commit_group()
                numerator = gl.amd.cdna4.mfma(
                    p, gl.convert_layout(value, rhs), numerators[panel] * alpha[:, None]
                )
                next_numerators += (numerator,)
            numerators = next_numerators
            if step + 1 < TILES:
                if next_active:
                    gl.barrier()
                    _prefetch_panel(
                        rotary, KV, next_slots, KV_ROW, KV_ROWS, 512, BLOCK_N
                    )
                    gl.amd.cdna4.async_copy.commit_group()
        if step + 1 < TILES:
            if next_active and (not active):
                gl.barrier()
                for panel in gl.static_range(5):
                    _prefetch_panel(
                        key_panels[panel],
                        KV,
                        next_slots,
                        KV_ROW,
                        KV_ROWS,
                        panel * 128,
                        BLOCK_N,
                    )
                    gl.amd.cdna4.async_copy.commit_group()
    return (numerators, maximum, denominator)


@gluon.jit
def _partial_attention(
    Q,
    KV,
    Slots,
    Partial,
    Stats,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    S_ROW: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    scale,
    BLOCK_N: gl.constexpr,
    TILES: gl.constexpr,
    ROW_GROUP: gl.constexpr,
):
    row, part = (gl.program_id(0), gl.program_id(1))
    if ROW_GROUP > 0:
        linear = row
        row = linear // (ROW_GROUP * SPLITS) * ROW_GROUP + linear % ROW_GROUP
        part = linear // ROW_GROUP % SPLITS
    q, qr = _load_query(Q, row, Q_ROW, Q_HEAD, HEADS, BLOCK_N)
    h = gl.arange(0, 16, gl.SliceLayout(1, _MMA))
    numerators, maximum, denominator = _pipeline_attention(
        KV,
        q,
        qr,
        Slots,
        row,
        part,
        KV_ROW,
        KV_ROWS,
        S_ROW,
        SPLITS,
        scale,
        BLOCK_N,
        TILES,
    )
    for panel in gl.static_range(4):
        _store_partial(
            numerators[panel], Partial, row, part, h[:, None], panel, HEADS, SPLITS
        )
    sh = gl.arange(0, 16, maximum.type.layout)
    stats_base = Stats + ((row * SPLITS + part) * HEADS + sh) * 2
    gl.store(stats_base, maximum, sh < HEADS)
    gl.store(stats_base + 1, denominator, sh < HEADS)


@gluon.jit
def _merge_statistics(Stats, row, h, HEADS: gl.constexpr, SPLITS: gl.constexpr):
    gl.static_assert(SPLITS == 2 or SPLITS == 4)
    maxima, denominators = ((), ())
    for part in gl.static_range(SPLITS):
        base = Stats + ((row * SPLITS + part) * HEADS + h) * 2
        maxima += (gl.load(base),)
        denominators += (gl.load(base + 1),)
    if SPLITS == 2:
        maximum = gl.maximum(maxima[0], maxima[1])
    else:
        maximum = gl.maximum(
            gl.maximum(maxima[0], maxima[2]), gl.maximum(maxima[1], maxima[3])
        )
    maximum = gl.where(maximum == -float("inf"), 0.0, maximum)
    weights, scaled_denominators = ((), ())
    for part in gl.static_range(SPLITS):
        weight = gl.exp2((maxima[part] - maximum) * _LOG2_E)
        weights += (weight,)
        scaled_denominators += (denominators[part] * weight,)
    if SPLITS == 2:
        denominator = scaled_denominators[0] + scaled_denominators[1]
    else:
        denominator = (
            scaled_denominators[0]
            + scaled_denominators[2]
            + (scaled_denominators[1] + scaled_denominators[3])
        )
    return (weights, denominator)


@gluon.jit
def _merge_native(Partial, Stats, Out, HEADS: gl.constexpr, SPLITS: gl.constexpr):
    if SPLITS == 2:
        layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]],
            lane_bases=[[1, 0], [2, 0], [4, 0], [0, 16], [0, 32], [0, 64]],
            warp_bases=[],
            block_bases=[],
            shape=[8, 128],
        )
        cache: gl.constexpr = ""
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [0, 1])
        cache: gl.constexpr = ".cg"
    row, panel = (gl.program_id(0), gl.program_id(1).to(gl.uint32))
    oh = gl.arange(0, 8, gl.SliceLayout(1, layout))
    weights, denominator = _merge_statistics(Stats, row, oh, HEADS, SPLITS)
    pd = gl.arange(0, 128, gl.SliceLayout(0, layout))
    partial_base = Partial + (row * 4 * SPLITS + panel * SPLITS) * HEADS * 128
    partial_offset = pd[None, :] // 8 * HEADS * 8 + oh[:, None] * 8 + pd[None, :] % 8
    acc = gl.zeros((8, 128), gl.float32, layout)
    for part in gl.static_range(SPLITS):
        weight = weights[part]
        p = gl.amd.cdna4.buffer_load(
            partial_base, part * HEADS * 128 + partial_offset, cache=cache
        ).to(gl.float32)
        acc = gl.fma(p, weight[:, None], acc)
    reciprocal = 1.0 / gl.where(denominator > 0.0, denominator, 1.0)
    result = acc * reciprocal[:, None]
    d = panel * 128 + pd // 8 * 4 + pd % 8 // 4 * 64 + pd % 4
    gl.store(
        Out + (row * HEADS + oh[:, None]) * 512 + d[None, :], result.to(gl.bfloat16)
    )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, heads, _ = query.shape
    selected = selected_slots.shape[1]
    assert heads == 8 and selected == 2048
    block_n = 128 if m >= 256 else 64
    tiles = 8
    splits = selected // (tiles * block_n)
    row_group = 4 if block_n == 64 and m % 4 == 0 else 0
    partial = torch.empty(
        (m, 4, splits, 16, heads, 8), device=query.device, dtype=torch.bfloat16
    )
    stats = torch.empty((m, splits, heads, 2), device=query.device, dtype=torch.float32)
    output = torch.empty((m, heads, 512), device=query.device, dtype=torch.bfloat16)
    grid = (m * splits,) if row_group else (m, splits)
    _partial_attention[grid](
        query,
        kv_cache,
        selected_slots,
        partial,
        stats,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        kv_cache.shape[0],
        selected_slots.stride(0),
        heads,
        splits,
        softmax_scale,
        block_n,
        tiles,
        row_group,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    _merge_native[m, 4](
        partial,
        stats,
        output,
        heads,
        splits,
        num_warps=1 if splits == 2 else 2,
        enable_fp_fusion=False,
    )
    return output
