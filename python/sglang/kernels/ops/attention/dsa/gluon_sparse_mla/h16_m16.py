# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m16.py
from math import prod

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _interleaved_partial_offset(
    row,
    head_tile,
    split,
    head,
    value,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    VALUE_MAJOR: gl.constexpr,
    RECORD_STRIDE: gl.constexpr,
    VALUE_GAP: gl.constexpr,
):
    if VALUE_MAJOR and VALUE_GAP:
        offset = (
            (row * gl.cdiv(HEADS, 16) + head_tile) * 128 * (SPLITS * 64 + VALUE_GAP)
            + value // 4 * (SPLITS * 64 + VALUE_GAP)
            + split * 64
            + head % 16 * 4
            + value % 4
        )
    elif VALUE_MAJOR:
        offset = (
            (row * gl.cdiv(HEADS, 16) + head_tile) * SPLITS * 8192
            + (value // 4 * SPLITS + split) * 64
            + head % 16 * 4
            + value % 4
        )
    else:
        offset = (
            ((row * gl.cdiv(HEADS, 16) + head_tile) * SPLITS + split) * RECORD_STRIDE
            + value // 4 * 64
            + head % 16 * 4
            + value % 4
        )
    return offset


@gluon.jit
def _stats_record(
    row,
    head_tile,
    split,
    head,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    SPLIT_MAJOR: gl.constexpr,
):
    if SPLIT_MAJOR:
        record = (
            (row * gl.cdiv(HEADS, 16) + head_tile) * SPLITS + split
        ) * 16 + head % 16
    else:
        record = (row * HEADS + head) * SPLITS + split
    return record


@gluon.jit
def _split_softmax(scores, active, scale):
    scores = gl.where(active[None, :], scores * scale, -float("inf"))
    maximum = gl.max(scores, 1)
    safe_maximum = gl.where(maximum == -float("inf"), 0.0, maximum)
    exponent = gl.exp2((scores - safe_maximum[:, None]) * 1.4426950408889634)
    probability = gl.where(active[None, :], exponent, 0.0)
    denominator = gl.sum(probability, 1)
    return (probability, maximum, denominator)


@gluon.jit
def _gather_words(
    KV,
    slots,
    active,
    K0: gl.constexpr,
    WIDTH: gl.constexpr,
    START: gl.constexpr,
    WAVES: gl.constexpr,
    ZERO_INVALID: gl.constexpr,
    BOUNDED_ZERO: gl.constexpr,
):
    word_layout: gl.constexpr = gl.BlockedLayout([4, 1], [4, 16], [1, WAVES], [0, 1])
    safe_slots = gl.convert_layout(
        gl.where(active, slots, 0), gl.SliceLayout(0, word_layout)
    )
    valid = gl.convert_layout(active, gl.SliceLayout(0, word_layout))
    d = gl.arange(0, WIDTH // 4, gl.SliceLayout(1, word_layout))
    if BOUNDED_ZERO:
        row_words = gl.where(valid, safe_slots.to(gl.int32) * (K0 // 4), 1 << 29)
        row_words = gl.multiple_of(row_words, 4)
        words = gl.amd.cdna4.buffer_load(
            KV.to(gl.pointer_type(gl.uint32)),
            row_words[None, :] + START // 4 + d[:, None],
        )
    else:
        words = gl.amd.cdna4.buffer_load(
            KV.to(gl.pointer_type(gl.uint32)),
            safe_slots[None, :].to(gl.int32) * (K0 // 4) + START // 4 + d[:, None],
        )
        if ZERO_INVALID:
            words = gl.where(valid[None, :], words, 0)
    return words


@gluon.jit
def _gather_bytes(KV, slots, active, channels, K0: gl.constexpr, START: gl.constexpr):
    row_bytes = gl.where(active, slots.to(gl.int32) * K0, -(1 << 31))
    row_bytes = gl.multiple_of(row_bytes, 16)
    return gl.amd.cdna4.buffer_load(KV, row_bytes[None, :] + START + channels[:, None])


@gluon.jit
def _query_key_scores(
    q_buffer,
    tiles_buffer,
    key_buffer,
    q_rope_buffer,
    key_rope_buffer,
    BLOCK_H: gl.constexpr,
    BLOCK_S: gl.constexpr,
    ROWS: gl.constexpr,
    MMA: gl.constexpr,
):
    dot_a: gl.constexpr = gl.DotOperandLayout(0, MMA, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, MMA, 8)
    scores = gl.zeros((BLOCK_H, BLOCK_S), gl.float32, MMA)
    if ROWS == 2:
        full_key = (
            tiles_buffer._reinterpret(
                dtype=key_buffer.dtype,
                shape=[4, BLOCK_S, 128],
                layout=gl.SwizzledSharedLayout(8, 1, 16, order=[2, 1, 0]),
            )
            .permute((0, 2, 1))
            .reshape((512, BLOCK_S))
        )
        q_operand = q_buffer.load(dot_a)
        key_operand = full_key.load(dot_b).to(gl.bfloat16)
        scores = gl.amd.cdna4.mfma(q_operand, key_operand, scores)
    else:
        for offset_k in gl.static_range(0, 512, 128):
            q_operand = q_buffer.slice(offset_k, 128, 1).load(dot_a)
            key_operand = (
                key_buffer.index(offset_k // 128)
                .permute((1, 0))
                .load(dot_b)
                .to(gl.bfloat16)
            )
            scores = gl.amd.cdna4.mfma(q_operand, key_operand, scores)
    q_rope_operand = q_rope_buffer.load(dot_a).to(gl.bfloat16)
    key_rope_operand = key_rope_buffer.load(dot_b).to(gl.bfloat16)
    return gl.amd.cdna4.mfma(q_rope_operand, key_rope_operand, scores)


@gluon.jit
def _attention_partials(
    Q,
    KV,
    Slots,
    Partial,
    Stats,
    scale,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    HEADS: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_S: gl.constexpr,
    BLOCK_V: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BUFFER_PARTIALS: gl.constexpr,
    SKIP_EMPTY: gl.constexpr,
    Q_LANES: gl.constexpr,
    Q_WAVES: gl.constexpr,
    INTERLEAVED: gl.constexpr,
    SPLIT_MAJOR_STATS: gl.constexpr,
    VALUE_MAJOR: gl.constexpr,
    WAVES: gl.constexpr,
    ROWS: gl.constexpr,
    PACKED_GATHER: gl.constexpr,
    BOUNDED_ZERO: gl.constexpr,
    RECORD_STRIDE: gl.constexpr,
    VALUE_GAP: gl.constexpr,
):
    MID_ROWS: gl.constexpr = ROWS > 4 and ROWS <= 8
    GATHER_KEY_LATE: gl.constexpr = ROWS > 2 and ROWS <= 8
    row = 0 if ROWS == 1 else gl.program_id(0)
    head_start = 0 if HEADS <= BLOCK_H else gl.program_id(1) * BLOCK_H
    head_tile = head_start // 16
    producer_sequence = (
        gl.program_id(2).to(gl.uint32) if ROWS == 1 else gl.program_id(2)
    )
    if MID_ROWS:
        split = producer_sequence % SPLITS
        value_start = producer_sequence // SPLITS * BLOCK_V
    else:
        split = producer_sequence // (512 // BLOCK_V)
        value_start = producer_sequence % (512 // BLOCK_V) * BLOCK_V
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WAVES]
    )
    pv_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=INTERLEAVED,
        warps_per_cta=[1, WAVES],
    )
    pv_a: gl.constexpr = gl.DotOperandLayout(0, pv_mma, 8)
    pv_b: gl.constexpr = gl.DotOperandLayout(1, pv_mma, 8)
    q_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // Q_LANES, Q_LANES], [WAVES // Q_WAVES, Q_WAVES], [1, 0]
    )
    k_layout: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, WAVES], [0, 1])
    query_vector: gl.constexpr = 16 if ROWS == 2 else 8
    query_per_phase: gl.constexpr = 1 if ROWS <= 4 else 2
    query_max_phase: gl.constexpr = 16 if ROWS == 1 else 8
    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        query_vector, query_per_phase, query_max_phase, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])
    hq = head_start + gl.arange(0, BLOCK_H, gl.SliceLayout(1, q_layout))
    dq = gl.arange(0, 512, gl.SliceLayout(0, q_layout))
    rope_layout: gl.constexpr = (
        gl.BlockedLayout([1, 8], [8, 8], [2, 4], [1, 0])
        if ROWS > 8 and WAVES == 8
        else q_layout
    )
    rope_heads = head_start + gl.arange(0, BLOCK_H, gl.SliceLayout(1, rope_layout))
    qr = gl.arange(0, 64, gl.SliceLayout(0, rope_layout))
    dk = gl.arange(0, 512, gl.SliceLayout(1, k_layout))
    kr = gl.arange(0, 64, gl.SliceLayout(1, k_layout))
    positions = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(0, k_layout))
    position_valid = (SELECTED % BLOCK_S == 0) | (positions < SELECTED)
    if ROWS > 8 or (ROWS > 2 and ROWS <= 4):
        slots = gl.amd.cdna4.buffer_load(
            Slots, row * S0 + positions, position_valid, -1
        ).to(gl.int64)
    else:
        slots = gl.load(Slots + row * S0 + positions, position_valid, -1).to(gl.int64)
    active = slots >= 0
    latent_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [BLOCK_H, 512], [1, 0]
    )
    q = gl.load(
        Q + row * Q0 + hq[:, None] * Q1 + dq[None, :],
        (HEADS % BLOCK_H == 0) | (hq[:, None] < HEADS),
        0.0,
    )
    if MID_ROWS:
        q_buffer = gl.allocate_shared_memory(q.dtype, q.shape, latent_shared, q)
    if not GATHER_KEY_LATE:
        if PACKED_GATHER:
            key = _gather_words(
                KV, slots, active, K0, 512, 0, WAVES, True, BOUNDED_ZERO
            )
        else:
            key = gl.load(KV + slots[None, :] * K0 + dk[:, None], active[None, :], 0.0)
    q_rope = gl.load(
        Q + row * Q0 + rope_heads[:, None] * Q1 + 512 + qr[None, :],
        (HEADS % BLOCK_H == 0) | (rope_heads[:, None] < HEADS),
        0.0,
    )
    if MID_ROWS:
        q_rope_buffer = gl.allocate_shared_memory(
            q_rope.dtype, q_rope.shape, shared_q, q_rope
        )
    if GATHER_KEY_LATE:
        if PACKED_GATHER:
            key = _gather_words(
                KV, slots, active, K0, 512, 0, WAVES, True, BOUNDED_ZERO
            )
        elif BOUNDED_ZERO:
            key = _gather_bytes(KV, slots, active, dk, K0, 0)
        else:
            key = gl.load(KV + slots[None, :] * K0 + dk[:, None], active[None, :], 0.0)
    if PACKED_GATHER:
        key_rope = _gather_words(
            KV, slots, active, K0, 64, 512, WAVES, False, BOUNDED_ZERO
        )
    elif BOUNDED_ZERO and ROWS <= 4:
        key_rope = _gather_bytes(KV, slots, active, kr, K0, 512)
    else:
        key_rope = gl.load(
            KV + slots[None, :] * K0 + 512 + kr[:, None], active[None, :], 0.0
        )
    if not MID_ROWS:
        q_buffer = gl.allocate_shared_memory(q.dtype, q.shape, latent_shared, q)
    PV_WIDTH: gl.constexpr = 128
    key_tiles = (
        key.permute((1, 0))
        .reshape(
            (BLOCK_S, 512 // PV_WIDTH, PV_WIDTH // 4 if PACKED_GATHER else PV_WIDTH)
        )
        .permute((1, 0, 2))
    )
    tile_phases: gl.constexpr = PV_WIDTH // 8
    tile_shared: gl.constexpr = gl.SwizzledSharedLayout(
        2 if PACKED_GATHER else 8, 1, tile_phases, order=[2, 1, 0]
    )
    tiles_buffer = gl.allocate_shared_memory(
        key.dtype, key_tiles.shape, tile_shared, key_tiles
    )
    key_buffer = tiles_buffer._reinterpret(
        dtype=KV.dtype.element_ty,
        shape=[512 // PV_WIDTH, BLOCK_S, PV_WIDTH],
        layout=gl.SwizzledSharedLayout(8, 1, tile_phases, order=[1, 0]),
    )
    if not MID_ROWS:
        q_rope_buffer = gl.allocate_shared_memory(
            q_rope.dtype, q_rope.shape, shared_q, q_rope
        )
    if PACKED_GATHER:
        rope_words_buffer = gl.allocate_shared_memory(
            gl.uint32,
            key_rope.shape,
            gl.SwizzledSharedLayout(4, 2, 8, order=[0, 1]),
            key_rope,
        )
        key_rope_buffer = rope_words_buffer._reinterpret(
            dtype=KV.dtype.element_ty, shape=[64, BLOCK_S], layout=shared_k
        )
    else:
        key_rope_buffer = gl.allocate_shared_memory(
            key_rope.dtype, key_rope.shape, shared_k, key_rope
        )
    scores = _query_key_scores(
        q_buffer,
        tiles_buffer,
        key_buffer,
        q_rope_buffer,
        key_rope_buffer,
        BLOCK_H,
        BLOCK_S,
        ROWS,
        mma,
    )
    if MID_ROWS:
        prefetched_raw_values = key_buffer.index(value_start // PV_WIDTH).load(pv_b)
    prob_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [BLOCK_H // WAVES, 64 * WAVES // BLOCK_H], [WAVES, 1], [1, 0]
    )
    score_phases: gl.constexpr = 8 if BLOCK_S == 64 else 16
    scores_buffer = gl.allocate_shared_memory(
        gl.float32,
        scores.shape,
        gl.SwizzledSharedLayout(4, 1, score_phases, [1, 0]),
        scores,
    )
    active_buffer = gl.allocate_shared_memory(
        gl.int32,
        active.shape,
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        active.to(gl.int32),
    )
    scores = scores_buffer.load(prob_layout)
    active_prob = active_buffer.load(gl.SliceLayout(0, prob_layout)) != 0
    if SKIP_EMPTY:
        has_values = gl.sum(active_prob.to(gl.int32), 0) != 0
    else:
        has_values = True
    probability, maximum, denominator = _split_softmax(scores, active_prob, scale)
    probability_high = probability.to(gl.bfloat16)
    probability_low = (probability - probability_high.to(gl.float32)).to(gl.bfloat16)
    probability_shared: gl.constexpr = (
        gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_S, 16]], [BLOCK_H, BLOCK_S], [1, 0]
        )
        if ROWS > 4
        else gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
        if ROWS == 2
        else shared_q
    )
    high_buffer = gl.allocate_shared_memory(
        gl.bfloat16, probability.shape, probability_shared, probability_high
    )
    low_buffer = gl.allocate_shared_memory(
        gl.bfloat16, probability.shape, probability_shared, probability_low
    )
    if ROWS > 8:
        first_values = (
            key_buffer.index(value_start // PV_WIDTH).load(pv_b).to(gl.bfloat16)
        )
    probability_hi = high_buffer.load(pv_a)
    probability_lo = low_buffer.load(pv_a)
    hm = head_start + gl.arange(0, BLOCK_H, gl.SliceLayout(1, pv_mma))
    for offset in gl.static_range(0, BLOCK_V, PV_WIDTH):
        if MID_ROWS and offset == 0:
            values = prefetched_raw_values.to(gl.bfloat16)
        elif ROWS > 8 and offset == 0:
            values = first_values
        else:
            values = (
                key_buffer.index(((value_start + offset) // PV_WIDTH).to(gl.int32))
                .load(pv_b)
                .to(gl.bfloat16)
            )
        numerator = gl.amd.cdna4.mfma(
            probability_lo, values, gl.zeros((BLOCK_H, PV_WIDTH), gl.float32, pv_mma)
        )
        numerator = gl.amd.cdna4.mfma(probability_hi, values, numerator)
        vm = value_start + offset + gl.arange(0, PV_WIDTH, gl.SliceLayout(0, pv_mma))
        if INTERLEAVED:
            partial_offset = _interleaved_partial_offset(
                row,
                head_tile,
                split,
                hm[:, None],
                vm[None, :],
                HEADS,
                SPLITS,
                VALUE_MAJOR,
                RECORD_STRIDE,
                VALUE_GAP,
            )
        else:
            local_v = gl.arange(0, PV_WIDTH, gl.SliceLayout(0, pv_mma))
            partial_offset = (
                (producer_sequence % 4 * SPLITS + split) * HEADS + hm[:, None]
            ) * 144 + local_v[None, :]
        if BUFFER_PARTIALS:
            gl.amd.cdna4.buffer_store(
                stored_value=numerator,
                ptr=Partial,
                offsets=partial_offset,
                mask=has_values & ((HEADS % BLOCK_H == 0) | (hm[:, None] < HEADS)),
            )
        else:
            gl.store(
                Partial + partial_offset,
                numerator,
                has_values & ((HEADS % BLOCK_H == 0) | (hm[:, None] < HEADS)),
                cache_modifier=".cs" if ROWS > 8 else "",
            )
    if value_start == 0:
        hs = head_start + gl.arange(0, BLOCK_H, gl.SliceLayout(1, prob_layout))
        stat_record = _stats_record(
            row, head_tile, split, hs, HEADS, SPLITS, SPLIT_MAJOR_STATS
        )
        gl.store(
            Stats + stat_record * 2, maximum, (HEADS % BLOCK_H == 0) | (hs < HEADS)
        )
        gl.store(
            Stats + stat_record * 2 + 1,
            denominator,
            (HEADS % BLOCK_H == 0) | (hs < HEADS),
        )
    q_buffer._keep_alive()
    q_rope_buffer._keep_alive()
    key_rope_buffer._keep_alive()
    scores_buffer._keep_alive()
    active_buffer._keep_alive()


@gluon.jit
def _merge_partials(
    Partial,
    Stats,
    Output,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_SPLITS: gl.constexpr,
    WIDTH: gl.constexpr,
    MERGE_H: gl.constexpr,
    INTERLEAVED: gl.constexpr,
    SPLIT_MAJOR_STATS: gl.constexpr,
    SKIP_EMPTY: gl.constexpr,
    RECIPROCAL: gl.constexpr,
    VALUE_MAJOR: gl.constexpr,
    RECORD_STRIDE: gl.constexpr,
    VALUE_GAP: gl.constexpr,
    PANEL_FIRST: gl.constexpr,
):
    if INTERLEAVED:
        row = gl.program_id(0).to(gl.uint32) if PANEL_FIRST else gl.program_id(0)
        first_head = 0 if HEADS <= MERGE_H else gl.program_id(1) * MERGE_H
        chunk = gl.program_id(2)
        if VALUE_MAJOR and PANEL_FIRST:
            head_groups: gl.constexpr = gl.cdiv(HEADS, MERGE_H)
            sequence = gl.program_id(1).to(gl.uint32) + head_groups * gl.program_id(
                2
            ).to(gl.uint32)
            first_head = sequence // 4 % head_groups * MERGE_H
            chunk = sequence % 4 * (128 // WIDTH) + sequence // (4 * head_groups)
    else:
        row = 0
        sequence = gl.program_id(0).to(gl.uint32) + HEADS * gl.program_id(1).to(
            gl.uint32
        )
        first_head = sequence // 4 % HEADS
        chunk = sequence % 4 * (128 // WIDTH) + sequence // (4 * HEADS)
    SPLIT_LANES: gl.constexpr = 4
    BUFFER_OUTPUT: gl.constexpr = INTERLEAVED and SPLIT_MAJOR_STATS and (MERGE_H == 16)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4 if INTERLEAVED else 2],
        [SPLIT_LANES, MERGE_H, 64 // SPLIT_LANES // MERGE_H],
        [1, 1, 1],
        [1, 2, 0],
    )
    stats_layout: gl.constexpr = gl.SliceLayout(2, layout)
    s = gl.arange(0, BLOCK_SPLITS, gl.SliceLayout(1, stats_layout))
    h = first_head + gl.arange(0, MERGE_H, gl.SliceLayout(0, stats_layout))
    v = chunk * WIDTH + gl.arange(
        0, WIDTH, gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    record = _stats_record(
        row, first_head // 16, s[:, None], h[None, :], HEADS, SPLITS, SPLIT_MAJOR_STATS
    )
    valid = (s[:, None] < SPLITS) & ((HEADS % MERGE_H == 0) | (h[None, :] < HEADS))
    if SPLIT_MAJOR_STATS or not INTERLEAVED:
        paired_stats = gl.amd.cdna4.buffer_load(
            Stats, gl.join(record * 2, record * 2 + 1), gl.join(valid, valid), 0.0
        )
        maximum, denominator = gl.split(paired_stats)
        maximum = gl.convert_layout(maximum, stats_layout, assert_trivial=True)
        denominator = gl.convert_layout(denominator, stats_layout, assert_trivial=True)
        maximum = gl.where(valid, maximum, -float("inf"))
    else:
        maximum = gl.amd.cdna4.buffer_load(Stats, record * 2, valid, -float("inf"))
        denominator = gl.amd.cdna4.buffer_load(Stats, record * 2 + 1, valid, 0.0)
    global_max = gl.max(maximum, 0)
    global_max = gl.where(global_max == -float("inf"), 0.0, global_max)
    correction = gl.exp2((maximum - global_max[None, :]) * 1.4426950408889634)
    total = gl.sum(denominator * correction, 0)
    if INTERLEAVED:
        partial_offset = _interleaved_partial_offset(
            row,
            first_head // 16,
            s[:, None, None],
            h[None, :, None],
            v[None, None, :],
            HEADS,
            SPLITS,
            VALUE_MAJOR,
            RECORD_STRIDE,
            VALUE_GAP,
        )
    else:
        partial_offset = (
            (v[None, None, :] // 128 * SPLITS + s[:, None, None]) * HEADS
            + h[None, :, None]
        ) * 144 + v[None, None, :] % 128
    if SKIP_EMPTY:
        partial_valid = valid & (denominator != 0)
    else:
        partial_valid = valid
    if BUFFER_OUTPUT and (not VALUE_MAJOR):
        partial = gl.load(
            Partial + partial_offset,
            partial_valid[:, :, None],
            0.0,
            cache_modifier=".cg",
        )
    else:
        partial = gl.amd.cdna4.buffer_load(
            Partial, partial_offset, partial_valid[:, :, None], 0.0
        )
    numerator = gl.sum(partial * correction[:, :, None], 0)
    output_heads: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(0, layout))
    v = gl.convert_layout(v, gl.SliceLayout(0, gl.SliceLayout(0, layout)))
    h = gl.convert_layout(h, output_heads)
    total = gl.convert_layout(total, output_heads)
    if RECIPROCAL:
        inverse_total = 1.0 / gl.where(total > 0, total, 1.0)
        result = numerator * inverse_total[:, None]
    else:
        result = numerator / gl.where(total[:, None] > 0, total[:, None], 1.0)
    if BUFFER_OUTPUT:
        gl.amd.cdna4.buffer_store(
            result.to(gl.bfloat16),
            Output,
            (row * HEADS + h[:, None]) * 512 + v[None, :],
            (HEADS % MERGE_H == 0) | (h[:, None] < HEADS),
        )
    else:
        gl.store(
            Output + (row * HEADS + h[:, None]) * 512 + v[None, :],
            result.to(gl.bfloat16),
            (HEADS % MERGE_H == 0) | (h[:, None] < HEADS),
        )


def _producer_geometry(rows):
    if rows <= 2:
        return (64, 128, 16, 4, 16, 1)
    if rows <= 4:
        return (64, 512, 8, 4, 8, 4)
    if rows <= 8:
        return (128, 256, 16, 8, 16, 2)
    return (128, 512, 16, 8, 32, 1)


def _merge_geometry(rows):
    if rows == 1:
        return (32, 1)
    if rows == 2 or 4 < rows <= 8:
        return (16, 4)
    return (8, 16 if rows > 8 else 8)


def _allocate_split_state(
    rows,
    heads,
    splits,
    device,
    interleaved,
    value_major,
    split_major_stats,
    record_stride,
    value_gap,
):
    if value_major:
        partial_shape = (rows, triton.cdiv(heads, 16), 128, splits * 64 + value_gap)
    elif interleaved:
        partial_shape = (rows, triton.cdiv(heads, 16), splits, record_stride)
    else:
        partial_shape = (4, splits, heads, 144)
    stats_heads = triton.cdiv(heads, 16) * 16 if split_major_stats else heads
    stats_shape = (rows, stats_heads, splits, 2)
    if rows > 8 or rows in (1, 2, 4, 8):
        partial_elements = prod(partial_shape)
        stats_elements = prod(stats_shape)
        gap = 32 if rows == 2 else 0
        arena = torch.empty(
            (partial_elements + stats_elements + gap,),
            dtype=torch.float32,
            device=device,
        )
        if rows == 2:
            stats = arena[:stats_elements].view(stats_shape)
            partial = arena[stats_elements + gap :].view(partial_shape)
        else:
            partial = arena[:partial_elements].view(partial_shape)
            stats = arena[partial_elements:].view(stats_shape)
    else:
        partial = torch.empty(partial_shape, dtype=torch.float32, device=device)
        stats = torch.empty(stats_shape, dtype=torch.float32, device=device)
    return (partial, stats)


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, heads, _ = query.shape
    selected = selected_slots.shape[1]
    bounded_zero = (
        (m <= 2 or 4 < m <= 8)
        and kv_cache.shape[0] > 0
        and (kv_cache.stride(0) % 16 == 0)
        and (kv_cache.storage_offset() % 16 == 0)
        and (kv_cache.shape[0] * kv_cache.stride(0) < 2**31 - 16)
    )
    packed_gather = (
        (m <= 2 or m > 8)
        and kv_cache.shape[0] > 0
        and (kv_cache.stride(0) % 4 == 0)
        and (kv_cache.storage_offset() % 4 == 0)
        and (kv_cache.shape[0] * kv_cache.stride(0) < 2**31)
        or bounded_zero
    )
    bounded_bytes = (
        2 < m <= 8
        and kv_cache.shape[0] > 0
        and (kv_cache.stride(0) > 0)
        and (kv_cache.stride(0) % 16 == 0)
        and (kv_cache.storage_offset() % 16 == 0)
        and (kv_cache.shape[0] * 576 < 2**31 - 16)
        and (kv_cache.shape[0] * kv_cache.stride(0) < 2**31 - 16)
        and ((kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576 < 2**31 - 16)
    )
    if bounded_bytes:
        packed_gather = False
        bounded_zero = True
    interleaved = m > 1
    value_major = m == 2 or 4 < m <= 8
    split_major_stats = 2 < m <= 4 or m > 8
    skip_empty = m <= 2 or split_major_stats
    buffer_partials = m <= 4
    block_s, block_v, block_h, waves, query_lanes, query_waves = _producer_geometry(m)
    value_gap = 16 if m == 2 else 0
    record_stride = 8224 if m > 8 else 8192
    splits = triton.cdiv(selected, block_s)
    output = torch.empty((m, heads, 512), dtype=torch.bfloat16, device=query.device)
    partial, stats = _allocate_split_state(
        m,
        heads,
        splits,
        query.device,
        interleaved,
        value_major,
        split_major_stats,
        record_stride,
        value_gap,
    )
    _attention_partials[m, triton.cdiv(heads, block_h), splits * (512 // block_v)](
        query,
        kv_cache,
        selected_slots,
        partial,
        stats,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        heads,
        selected,
        splits,
        block_s,
        block_v,
        block_h,
        buffer_partials,
        skip_empty,
        query_lanes,
        query_waves,
        interleaved,
        split_major_stats,
        value_major,
        waves,
        m,
        packed_gather,
        bounded_zero,
        record_stride,
        value_gap,
        num_warps=waves,
        enable_fp_fusion=False,
        waves_per_eu=3 if m <= 4 else 4 if m <= 8 else 0,
    )
    merge_width, merge_heads = _merge_geometry(m)
    if interleaved:
        merge_grid = (m, triton.cdiv(heads, merge_heads), 512 // merge_width)
    else:
        merge_grid = (m * heads, 16)
    _merge_partials[merge_grid](
        partial,
        stats,
        output,
        heads,
        splits,
        triton.next_power_of_2(splits),
        merge_width,
        merge_heads,
        interleaved,
        split_major_stats,
        skip_empty,
        interleaved and (m <= 4 or m > 8),
        value_major,
        record_stride,
        value_gap,
        m == 2,
        num_warps=1,
        enable_fp_fusion=m == 2 or m > 8,
    )
    return output
