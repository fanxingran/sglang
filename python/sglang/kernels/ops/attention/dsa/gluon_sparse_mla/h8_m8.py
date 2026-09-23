# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m8.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _stage_rotary_query(
    Q, row, group, Q0: gl.constexpr, Q1: gl.constexpr, H: gl.constexpr
):
    memory: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [4, 1], [1, 0])
    shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [8, 64], [1, 0]
    )
    heads = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, memory))
    d = gl.arange(0, 64, gl.SliceLayout(0, memory))
    tile = gl.allocate_shared_memory(gl.int16, [8, 64], shared)
    source = Q.to(gl.pointer_type(gl.int16))
    offsets = row * Q0 + heads[:, None] * Q1 + 512 + d[None, :]
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        tile, source, offsets.to(gl.int32), mask=(heads[:, None] < H) | (H % 8 == 0)
    )
    return tile


@gluon.jit
def _record_offsets(
    row,
    head,
    split,
    channel,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    PITCH: gl.constexpr,
):
    return ((row * H + head) * SPLITS + split) * PITCH + channel


@gluon.jit
def _store_statistics(
    Stats,
    Valid,
    row,
    group,
    split,
    value_tile,
    maximum,
    denominator,
    any_active,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    SCORE_ROWS: gl.constexpr,
    scores_layout: gl.constexpr,
    PARTIAL_PITCH: gl.constexpr,
    COLOCATED_STATS: gl.constexpr,
):
    physical_head = gl.arange(0, SCORE_ROWS, gl.SliceLayout(1, scores_layout))
    head = group * 8 + physical_head // (SCORE_ROWS // 8)
    record = (row * H + head) * SPLITS + split
    writer = (
        ((head < H) | (H % 8 == 0))
        & (physical_head % (SCORE_ROWS // 8) == 0)
        & (value_tile == 0)
    )
    if COLOCATED_STATS:
        gl.store(Stats + record * PARTIAL_PITCH, maximum, writer)
        gl.store(Stats + record * PARTIAL_PITCH + 1, denominator, writer)
        gl.store(Valid + record * PARTIAL_PITCH, any_active.to(gl.int32), writer)
    else:
        gl.store(Stats + record * 2, maximum, writer)
        gl.store(Stats + record * 2 + 1, denominator, writer)
        gl.store(
            Valid + row * SPLITS + split,
            any_active.to(gl.int32),
            (group == 0) & (value_tile == 0),
        )


@gluon.jit
def _split_attention(
    Q,
    KV,
    S,
    Partial,
    Stats,
    Valid,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    H: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_S: gl.constexpr,
    BLOCK_V: gl.constexpr,
    scale,
    QK_CHUNK: gl.constexpr,
    STAGE_SCORE: gl.constexpr,
    SCORE_PACK: gl.constexpr,
    PARTIAL_PITCH: gl.constexpr,
    COMPACT_SCORE: gl.constexpr,
    QUERY_TWO: gl.constexpr,
    PANEL_PARTIAL: gl.constexpr,
    PANEL_SIZE: gl.constexpr,
    ASYNC_KV: gl.constexpr,
    ASYNC_QUERY: gl.constexpr,
    BUFFER_QUERY: gl.constexpr,
    RESERVE_SCORE: gl.constexpr,
    COLOCATED_STATS: gl.constexpr,
    ASYNC_ROTARY: gl.constexpr,
    SPARSE_PARTIAL: gl.constexpr,
    STREAM_PV: gl.constexpr,
    WAVES: gl.constexpr,
    PRELOAD_INDICES: gl.constexpr,
    DIRECT_OFFSETS: gl.constexpr,
    VALUE_SELECTION: gl.constexpr = 0,
):
    row = gl.program_id(0)
    split = gl.program_id(2) if BLOCK_V <= 256 else gl.program_id(1)
    shard = gl.program_id(1) if BLOCK_V <= 256 else gl.program_id(2)
    group = 0
    value_tile = shard % (512 // BLOCK_V)
    if ASYNC_KV:
        memory: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0]]
            + ([[4, 0]] if WAVES == 2 else [])
            + ([[32, 0]] if BLOCK_S >= 64 else [])
            + ([[64, 0]] if BLOCK_S >= 128 else []),
            lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
            warp_bases=[[8, 0]] if WAVES == 2 else [[4, 0], [8, 0]],
            block_bases=[],
            shape=[BLOCK_S, 512],
        )
    else:
        memory: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    if ASYNC_QUERY:
        query_memory: gl.constexpr = gl.BlockedLayout(
            [1, 8], [1, 64], [WAVES, 1], [1, 0]
        )
    else:
        query_memory: gl.constexpr = gl.BlockedLayout(
            [1, 16], [4, 16], [2, 2] if QUERY_TWO else [4, 1], [1, 0]
        )
    qk_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WAVES]
    )
    qk_a: gl.constexpr = gl.DotOperandLayout(0, qk_matrix, 8)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, qk_matrix, 8)
    pv_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, WAVES]
    )
    pv_a: gl.constexpr = gl.DotOperandLayout(0, pv_matrix, 8)
    pv_b: gl.constexpr = gl.DotOperandLayout(1, pv_matrix, 8)
    scores_layout: gl.constexpr = gl.BlockedLayout(
        [1, SCORE_PACK], [4, 16], [WAVES, 1], [1, 0]
    )
    operand_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
    if BLOCK_S == 64 and BLOCK_V == 512:
        probability_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[64, 8]], [16, 64], [1, 0]
        )
    else:
        probability_layout: gl.constexpr = operand_shared
    if ASYNC_KV:
        latent_layout: gl.constexpr = gl.PaddedSharedLayout(
            [[1024, 16]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
                [16, 0],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
            ]
            + ([[32, 0]] if BLOCK_S >= 64 else [])
            + ([[64, 0]] if BLOCK_S >= 128 else []),
            [],
            [BLOCK_S, 512],
        )
    else:
        latent_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[512, 16]], [BLOCK_S, 512], [1, 0]
        )
    EARLY_QUERY: gl.constexpr = ASYNC_ROTARY and BLOCK_S == 64
    SCORE_ROWS: gl.constexpr = 8 if COMPACT_SCORE else 16
    query_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [8, 512], [1, 0]
    )
    if RESERVE_SCORE and EARLY_QUERY:
        score_shared = gl.allocate_shared_memory(
            gl.float32, [SCORE_ROWS, BLOCK_S], operand_shared
        )
    heads = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, query_memory))
    query_d = gl.arange(0, 512, gl.SliceLayout(0, query_memory))
    if ASYNC_QUERY and EARLY_QUERY:
        query_shared = gl.allocate_shared_memory(
            Q.dtype.element_ty, [8, 512], query_layout
        )
        query_offsets = row * Q0 + heads[:, None] * Q1 + query_d[None, :]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            query_shared,
            Q,
            query_offsets.to(gl.int32),
            mask=(heads[:, None] < H) | (H % 8 == 0),
        )
    d = gl.arange(0, 512, gl.SliceLayout(0, memory))
    pos = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(1, memory))
    in_bounds = (pos < SELECTED) | (SELECTED % BLOCK_S == 0)
    slots = gl.load(S + row * S0 + pos, in_bounds, -1).to(gl.int64)
    active = in_bounds & (slots >= 0)
    if PRELOAD_INDICES:
        rope_pos = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(0, qk_b))
        rope_in_bounds = (rope_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        rope_slots = gl.load(S + row * S0 + rope_pos, rope_in_bounds, -1).to(gl.int64)
        rope_active = rope_in_bounds & (rope_slots >= 0)
        score_pos = split * BLOCK_S + gl.arange(
            0, BLOCK_S, gl.SliceLayout(0, scores_layout)
        )
        score_in_bounds = (score_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        score_slots = gl.load(S + row * S0 + score_pos, score_in_bounds, -1)
        active_scores = score_in_bounds & (score_slots >= 0)
        any_active = gl.sum(active_scores.to(gl.int32), 0) > 0
    if RESERVE_SCORE and (not EARLY_QUERY):
        score_shared = gl.allocate_shared_memory(
            gl.float32, [SCORE_ROWS, BLOCK_S], operand_shared
        )
    if ASYNC_QUERY and (not EARLY_QUERY):
        query_shared = gl.allocate_shared_memory(
            Q.dtype.element_ty, [8, 512], query_layout
        )
        query_offsets = row * Q0 + heads[:, None] * Q1 + query_d[None, :]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            query_shared,
            Q,
            query_offsets.to(gl.int32),
            mask=(heads[:, None] < H) | (H % 8 == 0),
        )
    if ASYNC_KV:
        latent_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty, [BLOCK_S, 512], latent_layout
        )
        if DIRECT_OFFSETS:
            slot_offsets = slots.to(gl.int32) * K0
        else:
            slot_offsets = gl.where(active, slots, 0).to(gl.int32) * K0
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            latent_shared, KV, slot_offsets[:, None] + d[None, :], mask=active[:, None]
        )
        if ASYNC_ROTARY:
            rotary_query_shared = _stage_rotary_query(Q, row, group, Q0, Q1, H)
        gl.amd.cdna4.async_copy.commit_group()
    if not ASYNC_KV and (not PRELOAD_INDICES):
        score_pos = split * BLOCK_S + gl.arange(
            0, BLOCK_S, gl.SliceLayout(0, scores_layout)
        )
        score_in_bounds = (score_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        score_slots = gl.load(S + row * S0 + score_pos, score_in_bounds, -1)
        active_scores = score_in_bounds & (score_slots >= 0)
        any_active = gl.sum(active_scores.to(gl.int32), 0) > 0
    if not ASYNC_KV:
        slot_offsets = gl.where(active, slots, 0).to(gl.int32) * K0
        latent = gl.amd.cdna4.buffer_load(
            KV, slot_offsets[:, None] + d[None, :], active[:, None], 0.0
        )
    if not ASYNC_QUERY:
        q = gl.load(
            Q + row * Q0 + heads[:, None] * Q1 + query_d[None, :],
            (heads[:, None] < H) | (H % 8 == 0),
            0.0,
        )
    if not ASYNC_ROTARY:
        rope_heads = group * 8 + gl.arange(0, SCORE_ROWS, gl.SliceLayout(1, qk_a)) // (
            SCORE_ROWS // 8
        )
        rope_qd = gl.arange(0, 64, gl.SliceLayout(0, qk_a))
        if BUFFER_QUERY:
            rope_qoffsets = row * Q0 + rope_heads[:, None] * Q1 + 512 + rope_qd[None, :]
            rotary_queries = gl.amd.cdna4.buffer_load(
                Q,
                rope_qoffsets.to(gl.int32),
                (rope_heads[:, None] < H) | (H % 8 == 0),
                0.0,
            )
        else:
            rotary_queries = gl.load(
                Q + row * Q0 + rope_heads[:, None] * Q1 + 512 + rope_qd[None, :],
                (rope_heads[:, None] < H) | (H % 8 == 0),
                0.0,
            )
    if not PRELOAD_INDICES:
        rope_pos = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(0, qk_b))
        rope_in_bounds = (rope_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        rope_slots = gl.load(S + row * S0 + rope_pos, rope_in_bounds, -1).to(gl.int64)
        rope_active = rope_in_bounds & (rope_slots >= 0)
    rope_kd = gl.arange(0, 64, gl.SliceLayout(1, qk_b))
    if DIRECT_OFFSETS:
        rope_offsets = rope_slots.to(gl.int32) * K0
    else:
        rope_offsets = gl.where(rope_active, rope_slots, 0).to(gl.int32) * K0
    rotary_keys = gl.amd.cdna4.buffer_load(
        KV, rope_offsets[None, :] + 512 + rope_kd[:, None], rope_active[None, :], 0.0
    ).to(gl.bfloat16)
    if BLOCK_S < 128:
        probability_shared = gl.allocate_shared_memory(
            gl.bfloat16, [16, BLOCK_S], probability_layout
        )
    if not ASYNC_QUERY:
        query_shared = gl.allocate_shared_memory(q.dtype, q.shape, query_layout, q)
    if ASYNC_KV:
        gl.amd.cdna4.async_copy.wait_group(0)
    else:
        latent_shared = gl.allocate_shared_memory(
            latent.dtype, [BLOCK_S, 512], latent_layout, latent
        )
    if ASYNC_KV and (not PRELOAD_INDICES):
        score_pos = split * BLOCK_S + gl.arange(
            0, BLOCK_S, gl.SliceLayout(0, scores_layout)
        )
        score_in_bounds = (score_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        score_slots = gl.load(S + row * S0 + score_pos, score_in_bounds, -1)
        active_scores = score_in_bounds & (score_slots >= 0)
        any_active = gl.sum(active_scores.to(gl.int32), 0) > 0
    score = gl.full((SCORE_ROWS, BLOCK_S), 0, gl.float32, qk_matrix)
    for chunk in gl.static_range(0, 512 // QK_CHUNK):
        if not COMPACT_SCORE:
            gather_head = gl.arange(0, 16, gl.SliceLayout(1, qk_a)) // 2
            query_indices = gather_head[:, None] + gl.full(
                (16, QK_CHUNK), 0, gl.int32, qk_a
            )
            queries = query_shared.slice(chunk * QK_CHUNK, QK_CHUNK, 1).gather(
                query_indices, 0
            )
        else:
            queries = query_shared.slice(chunk * QK_CHUNK, QK_CHUNK, 1).load(qk_a)
        keys = (
            latent_shared.slice(chunk * QK_CHUNK, QK_CHUNK, 1)
            .permute([1, 0])
            .load(qk_b)
            .to(gl.bfloat16)
        )
        score = gl.amd.cdna4.mfma(queries, keys, score)
    if ASYNC_ROTARY:
        gl.static_assert(COMPACT_SCORE)
        rotary_queries = rotary_query_shared.load(qk_a).to(gl.bfloat16, bitcast=True)
    score = gl.amd.cdna4.mfma(rotary_queries, rotary_keys, score)
    if STAGE_SCORE:
        if RESERVE_SCORE:
            score_shared.store(score)
        else:
            score_shared = gl.allocate_shared_memory(
                gl.float32, [SCORE_ROWS, BLOCK_S], operand_shared, score
            )
        score = score_shared.load(scores_layout) * scale
    else:
        score = gl.convert_layout(score, scores_layout) * scale
    score = gl.where(active_scores[None, :], score, -float("inf"))
    maximum = gl.maximum(-float("inf"), gl.max(score, 1))
    safe_max = gl.where(any_active, maximum, 0.0)
    probability = gl.where(
        active_scores[None, :],
        gl.exp2((score - safe_max[:, None]) * 1.4426950408889634),
        0.0,
    )
    denominator = gl.sum(probability, 1)
    probability_hi = probability.to(gl.bfloat16)
    probability_lo = (probability - probability_hi.to(gl.float32)).to(gl.bfloat16)
    physical_head = gl.arange(0, SCORE_ROWS, gl.SliceLayout(1, scores_layout))
    if COMPACT_SCORE:
        paired_probability = gl.reshape(
            gl.permute(gl.join(probability_hi, probability_lo), (0, 2, 1)),
            (16, BLOCK_S),
        )
    else:
        paired_probability = gl.where(
            physical_head[:, None] % 2 == 0, probability_hi, probability_lo
        )
    if BLOCK_S < 128:
        probability_shared.store(paired_probability)
    else:
        probability_shared = gl.allocate_shared_memory(
            gl.bfloat16,
            paired_probability.shape,
            probability_layout,
            paired_probability,
        )
    probability_first: gl.constexpr = BLOCK_V == 512 and BLOCK_S < 128
    if probability_first:
        probabilities = probability_shared.load(pv_a)
    if VALUE_SELECTION:
        gl.static_assert(BLOCK_V == 128)
        if VALUE_SELECTION == 2:
            if value_tile < 2:
                half_values = latent_shared.slice(0, 256, 1)
            else:
                half_values = latent_shared.slice(256, 256, 1)
            if value_tile % 2 == 0:
                value_shared = half_values.slice(0, 128, 1)
            else:
                value_shared = half_values.slice(128, 128, 1)
        elif value_tile == 0:
            value_shared = latent_shared.slice(0, 128, 1)
        elif value_tile == 1:
            value_shared = latent_shared.slice(128, 128, 1)
        elif value_tile == 2:
            value_shared = latent_shared.slice(256, 128, 1)
        else:
            value_shared = latent_shared.slice(384, 128, 1)
    PV_WIDTH: gl.constexpr = 256 if STREAM_PV else BLOCK_V
    for fragment in gl.static_range(BLOCK_V // PV_WIDTH):
        if VALUE_SELECTION:
            values = value_shared.load(pv_b).to(gl.bfloat16)
        elif BLOCK_V == 512:
            values = (
                latent_shared.slice(fragment * PV_WIDTH, PV_WIDTH, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        elif BLOCK_V == 256:
            if value_tile == 0:
                values = latent_shared.slice(0, 256, 1).load(pv_b).to(gl.bfloat16)
            else:
                values = latent_shared.slice(256, 256, 1).load(pv_b).to(gl.bfloat16)
        elif value_tile == 0:
            values = latent_shared.slice(0, 128, 1).load(pv_b).to(gl.bfloat16)
        elif value_tile == 1:
            values = latent_shared.slice(128, 128, 1).load(pv_b).to(gl.bfloat16)
        elif value_tile == 2:
            values = latent_shared.slice(256, 128, 1).load(pv_b).to(gl.bfloat16)
        else:
            values = latent_shared.slice(384, 128, 1).load(pv_b).to(gl.bfloat16)
        if not probability_first:
            probabilities = probability_shared.load(pv_a)
        paired_numerator = gl.full((16, PV_WIDTH), 0, gl.float32, pv_matrix)
        paired_numerator = gl.amd.cdna4.mfma(probabilities, values, paired_numerator)
        numerator = gl.sum(gl.reshape(paired_numerator, (8, 2, PV_WIDTH)), 1)
        out_head = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, numerator.type.layout))
        out_d = (
            value_tile * BLOCK_V
            + fragment * PV_WIDTH
            + gl.arange(0, PV_WIDTH, gl.SliceLayout(0, numerator.type.layout))
        )
        if PANEL_PARTIAL:
            offsets = (
                (row * H + out_head[:, None]) * SPLITS * 512
                + out_d[None, :] // PANEL_SIZE * SPLITS * PANEL_SIZE
                + split * PANEL_SIZE
                + out_d[None, :] % PANEL_SIZE
            )
            gl.amd.cdna4.buffer_store(
                ptr=Partial,
                offsets=offsets.to(gl.int32),
                stored_value=numerator,
                mask=((out_head[:, None] < H) | (H % 8 == 0))
                & ((not SPARSE_PARTIAL) | any_active),
            )
        elif BLOCK_S == 128:
            offsets = _record_offsets(
                row, out_head[:, None], split, out_d[None, :], H, SPLITS, PARTIAL_PITCH
            )
            gl.amd.cdna4.buffer_store(
                ptr=Partial,
                offsets=offsets.to(gl.int32),
                stored_value=numerator,
                mask=((out_head[:, None] < H) | (H % 8 == 0))
                & ((not SPARSE_PARTIAL) | any_active),
            )
        else:
            records = (row * H + out_head) * SPLITS + split
            gl.store(
                Partial + records[:, None] * PARTIAL_PITCH + out_d[None, :],
                numerator,
                ((out_head[:, None] < H) | (H % 8 == 0))
                & ((not SPARSE_PARTIAL) | any_active),
            )
    _store_statistics(
        Stats,
        Valid,
        row,
        group,
        split,
        value_tile,
        maximum,
        denominator,
        any_active,
        H,
        SPLITS,
        SCORE_ROWS,
        scores_layout,
        PARTIAL_PITCH,
        COLOCATED_STATS,
    )


@gluon.jit
def _merge_attention(
    Partial,
    Stats,
    Valid,
    O,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    REDUCTION_WIDTH: gl.constexpr,
    BLOCK_D: gl.constexpr,
    MERGE_WARPS: gl.constexpr,
    PARTIAL_PITCH: gl.constexpr,
    PANEL_PARTIAL: gl.constexpr,
    PANEL_SIZE: gl.constexpr,
    SIMPLE_MAX: gl.constexpr,
    COLOCATED_STATS: gl.constexpr,
    SPARSE_PARTIAL: gl.constexpr,
):
    row = gl.program_id(0)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    wave_channels: gl.constexpr = BLOCK_D // MERGE_WARPS
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [256 // wave_channels, wave_channels // 4], [1, MERGE_WARPS], [1, 0]
    )
    s = gl.arange(0, REDUCTION_WIDTH, gl.SliceLayout(1, layout))
    d = tile * BLOCK_D + gl.arange(0, BLOCK_D, gl.SliceLayout(0, layout))
    record = (row * H + head) * SPLITS + s
    if COLOCATED_STATS:
        maximum = gl.load(Stats + record * PARTIAL_PITCH, s < SPLITS, -float("inf"))
        denominator = gl.load(Stats + record * PARTIAL_PITCH + 1, s < SPLITS, 0)
        count = gl.load(Valid + record * PARTIAL_PITCH, s < SPLITS, 0)
    else:
        maximum = gl.load(Stats + record * 2, s < SPLITS, -float("inf"))
        denominator = gl.load(Stats + record * 2 + 1, s < SPLITS, 0)
        count = gl.load(Valid + row * SPLITS + s, s < SPLITS, 0)
    global_max = gl.maximum(-float("inf"), gl.max(maximum, 0))
    if not SIMPLE_MAX:
        global_max = gl.where(gl.sum(count, 0) > 0, global_max, 0.0)
    factor = gl.where(
        count > 0, gl.exp2((maximum - global_max) * 1.4426950408889634), 0.0
    )
    if PANEL_PARTIAL:
        offsets = (
            (row * H + head) * SPLITS * 512
            + d[None, :] // PANEL_SIZE * SPLITS * PANEL_SIZE
            + s[:, None] * PANEL_SIZE
            + d[None, :] % PANEL_SIZE
        )
        numerators = gl.amd.cdna4.buffer_load(
            Partial,
            offsets.to(gl.int32),
            (s[:, None] < SPLITS) & ((not SPARSE_PARTIAL) | (count[:, None] > 0)),
            0.0,
        )
    else:
        numerators = gl.load(
            Partial + record[:, None] * PARTIAL_PITCH + d[None, :],
            (s[:, None] < SPLITS) & ((not SPARSE_PARTIAL) | (count[:, None] > 0)),
            0.0,
        )
    numerator = gl.sum(numerators * factor[:, None], 0)
    denominator = gl.sum(denominator * factor, 0)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    result = numerator * inverse_denominator
    gl.store(O + (row * H + head) * 512 + d, result.to(gl.bfloat16))


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, h, d = query.shape
    assert m in (1, 2, 3, 4, 7, 8, 15, 16) and h == 8 and (d == 576)
    assert query.dtype == torch.bfloat16
    assert kv_cache.ndim == 2 and kv_cache.shape[1] == 576
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert selected_slots.ndim == 2 and selected_slots.shape[0] == m
    assert selected_slots.shape[1] == 2048
    assert selected_slots.dtype in (torch.int32, torch.int64)
    assert query.stride(-1) == kv_cache.stride(-1) == selected_slots.stride(-1) == 1
    assert (
        query.device == kv_cache.device == selected_slots.device and softmax_scale > 0
    )
    block_size = 32 if m == 1 else 64 if m < 16 else 128
    value_width = 128 if m <= 2 else 256 if m <= 4 else 512
    splits = triton.cdiv(selected_slots.shape[1], block_size)
    cache_span = (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576
    assert kv_cache.stride(0) >= 0 and cache_span < 2**31
    query_span = (m - 1) * query.stride(0) + (h - 1) * query.stride(1) + 576
    assert min(query.stride(0), query.stride(1)) >= 0 and query_span * 2 < 2**31
    assert query.storage_offset() % 8 == 0 and kv_cache.storage_offset() % 16 == 0
    assert query.stride(0) % 8 == 0 and query.stride(1) % 8 == 0
    async_kv = m in (1, 2, 4, 8, 16) and kv_cache.stride(0) == 576
    async_query = async_kv
    async_rotary = m in (8, 16) and async_kv
    sparse_partial = m == 8 and async_query
    stream_pv = m == 8 and async_query
    qk_chunk = 512 if m <= 8 and async_kv else 128
    score_pack = 2 if m == 1 else 4 if m < 16 else 8
    producer_waves = 2 if m == 1 and async_query else 4
    compact_score = (
        producer_waves == 2
        or (1 < m <= 4 and (not async_query))
        or async_rotary
        or (m == 16)
    )
    stage_score = async_rotary or (
        m < 16 if async_query else not (2 < m <= 4 or m == 16)
    )
    output = torch.empty((m, h, 512), device=query.device, dtype=torch.bfloat16)
    panel_partial = m <= 2 and m * h * splits * 512 * 4 < 2**31
    colocated_stats = m == 16
    partial_pitch = 528 if colocated_stats else 512
    panel_size = 32 if m == 2 else 16
    if colocated_stats:
        partial = torch.empty(
            (m, h, splits, partial_pitch), device=query.device, dtype=torch.float32
        )
        stats = partial[..., 512:514]
        valid = partial.view(torch.int32)[..., 514:515]
    else:
        partial_words = m * h * splits * partial_pitch
        stats_words = m * h * splits * 2
        valid_words = m * splits
        storage = torch.empty(
            partial_words + stats_words + valid_words,
            device=query.device,
            dtype=torch.float32,
        )
        partial = storage[:partial_words]
        stats = storage[partial_words : partial_words + stats_words]
        valid = storage[partial_words + stats_words :].view(torch.int32)
    shards = triton.cdiv(h, 8) * (512 // value_width)
    grid = (m, shards, splits) if value_width <= 256 else (m, splits, shards)
    _split_attention[grid](
        query,
        kv_cache,
        selected_slots,
        partial,
        stats,
        valid,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        h,
        selected_slots.shape[1],
        splits,
        block_size,
        value_width,
        softmax_scale,
        qk_chunk,
        stage_score,
        score_pack,
        partial_pitch,
        compact_score,
        m > 4,
        panel_partial,
        panel_size,
        async_kv,
        async_query,
        m <= 2 and async_query or m == 8 or m == 16,
        async_query,
        colocated_stats,
        async_rotary,
        sparse_partial,
        stream_pv,
        WAVES=producer_waves,
        PRELOAD_INDICES=m == 4 and async_query,
        DIRECT_OFFSETS=m in (2, 8) and async_query,
        VALUE_SELECTION=(2 if m == 1 else 1) if m <= 2 and async_query else 0,
        num_warps=producer_waves,
        enable_fp_fusion=False,
        llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]
        if async_kv and m == 16
        else [],
    )
    merge_width = (
        16 if m == 1 else 32 if m == 2 else 64 if m <= 4 else 128 if m < 16 else 256
    )
    merge_warps = 1 if m <= 2 else 2 if m <= 4 else 4
    _merge_attention[m, h, 512 // merge_width](
        partial,
        stats,
        valid,
        output,
        h,
        splits,
        triton.next_power_of_2(splits),
        merge_width,
        merge_warps,
        partial_pitch,
        panel_partial,
        panel_size,
        m <= 2,
        colocated_stats,
        sparse_partial,
        num_warps=merge_warps,
        enable_fp_fusion=False,
    )
    return output
