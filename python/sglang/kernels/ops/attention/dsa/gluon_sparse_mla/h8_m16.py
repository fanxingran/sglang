# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m16.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


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
    BUFFER_KV: gl.constexpr,
    STAGE_SCORE: gl.constexpr,
    SCORE_PACK: gl.constexpr,
    LATENT_PAD: gl.constexpr,
    SHARD_FIRST: gl.constexpr,
    NATIVE_SCORE: gl.constexpr,
    BUFFER_PARTIAL: gl.constexpr,
    QUERY_WARPS: gl.constexpr,
    EARLY_PROB: gl.constexpr,
    WIDE_QUERY: gl.constexpr,
    WIDE_SLOTS: gl.constexpr,
    WIDE_PARTIAL: gl.constexpr,
    PARTIAL_STRIDE: gl.constexpr,
    ASYNC_KV: gl.constexpr,
    COLOCATED: gl.constexpr,
    ASYNC_QUERY: gl.constexpr,
    QK_WIDTH: gl.constexpr,
    PAIR_SLOT: gl.constexpr,
    DIRECT_OFFSETS: gl.constexpr,
    PV_FRAGMENT: gl.constexpr,
    STREAM_VALUES: gl.constexpr,
    PRODUCER_WARPS: gl.constexpr = 4,
    ASYNC_ROPE: gl.constexpr = False,
    RESERVE_PROB: gl.constexpr = False,
):
    row = gl.program_id(0)
    if SHARD_FIRST:
        split = gl.program_id(2)
        shard = gl.program_id(1)
    else:
        split = gl.program_id(1)
        shard = gl.program_id(2)
    if H <= 8:
        group = 0
    else:
        group = shard // (512 // BLOCK_V)
    value_tile = shard % (512 // BLOCK_V)
    query_row = row.to(gl.int64) if WIDE_QUERY else row
    slot_row = row.to(gl.int64) if WIDE_SLOTS else row
    partial_row = row.to(gl.int64) if WIDE_PARTIAL else row
    memory: gl.constexpr = gl.BlockedLayout(
        [1, 8 if BLOCK_S == 32 else 16], [4, 16], [PRODUCER_WARPS, 1], [1, 0]
    )
    qk_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, PRODUCER_WARPS],
    )
    qk_pack: gl.constexpr = 16 if BLOCK_S == 128 and ASYNC_ROPE else 8
    qk_a: gl.constexpr = gl.DotOperandLayout(0, qk_matrix, qk_pack)
    qk_b: gl.constexpr = gl.DotOperandLayout(1, qk_matrix, qk_pack)
    pv_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[1, PRODUCER_WARPS],
    )
    pv_a: gl.constexpr = gl.DotOperandLayout(0, pv_matrix, 8)
    pv_b: gl.constexpr = gl.DotOperandLayout(1, pv_matrix, 8)
    if NATIVE_SCORE:
        scores_layout: gl.constexpr = gl.BlockedLayout(
            [1, SCORE_PACK], [2, 32], [PRODUCER_WARPS, 1], [1, 0]
        )
    else:
        scores_layout: gl.constexpr = gl.BlockedLayout(
            [1, SCORE_PACK], [4, 16], [PRODUCER_WARPS, 1], [1, 0]
        )
    operand_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
    if RESERVE_PROB:
        probability_shared = gl.allocate_shared_memory(
            gl.bfloat16, [16, BLOCK_S], operand_shared
        )
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
                [PAIR_SLOT, 0],
                [1, 0],
                [2, 0],
            ]
            + ([[4, 0]] if PAIR_SLOT != 4 else [])
            + [[8, 0]]
            + ([[16, 0]] if PAIR_SLOT != 16 else [])
            + ([[32, 0]] if BLOCK_S >= 64 else [])
            + ([[64, 0]] if BLOCK_S >= 128 else []),
            [],
            [BLOCK_S, 512],
        )
        copy_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0]]
            + ([[16 if PAIR_SLOT == 4 else 8, 0]] if PRODUCER_WARPS == 2 else [])
            + ([[32, 0]] if BLOCK_S >= 64 else [])
            + ([[64, 0]] if BLOCK_S >= 128 else []),
            lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [PAIR_SLOT, 0]],
            warp_bases=([[8, 0], [16, 0]] if PAIR_SLOT == 4 else [[4, 0], [8, 0]])[
                : 1 if PRODUCER_WARPS == 2 else 2
            ],
            block_bases=[],
            shape=[BLOCK_S, 512],
        )
    else:
        latent_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[512, LATENT_PAD]], [BLOCK_S, 512], [1, 0]
        )
    query_memory: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 16], [QUERY_WARPS, PRODUCER_WARPS // QUERY_WARPS], [1, 0]
    )
    heads = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, query_memory))
    query_d = gl.arange(0, 512, gl.SliceLayout(0, query_memory))
    if WIDE_QUERY:
        heads = heads.to(gl.int64)
    d = gl.arange(0, 512, gl.SliceLayout(0, memory))
    pos = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(1, memory))
    in_bounds = (pos < SELECTED) | (SELECTED % BLOCK_S == 0)
    slots = gl.load(S + slot_row * S0 + pos, in_bounds, -1).to(gl.int64)
    active = in_bounds & (slots >= 0)
    score_pos = split * BLOCK_S + gl.arange(
        0, BLOCK_S, gl.SliceLayout(0, scores_layout)
    )
    score_in_bounds = (score_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
    score_slots = gl.load(S + slot_row * S0 + score_pos, score_in_bounds, -1)
    active_scores = score_in_bounds & (score_slots >= 0)
    any_active = gl.sum(active_scores.to(gl.int32), 0) > 0
    query_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 16]], [8, 512], [1, 0]
    )
    if ASYNC_QUERY:
        query_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8], [1, 64], [PRODUCER_WARPS, 1], [1, 0]
        )
        copy_heads = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, query_copy_layout))
        copy_query_d = gl.arange(0, 512, gl.SliceLayout(0, query_copy_layout))
        query_shared = gl.allocate_shared_memory(
            Q.dtype.element_ty, [8, 512], query_layout
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            query_shared,
            Q,
            row * Q0 + copy_heads[:, None] * Q1 + copy_query_d[None, :],
            mask=(copy_heads[:, None] < H) | (H % 8 == 0),
        )
    if ASYNC_ROPE:
        rope_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2], [2, 32], [PRODUCER_WARPS, 1], [1, 0]
        )
        rope_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[128, 8]], [8, 64], [1, 0]
        )
        rope_copy_heads = group * 8 + gl.arange(
            0, 8, gl.SliceLayout(1, rope_copy_layout)
        )
        rope_copy_d = gl.arange(0, 64, gl.SliceLayout(0, rope_copy_layout))
        rotary_query_shared = gl.allocate_shared_memory(
            Q.dtype.element_ty, [8, 64], rope_shared_layout
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            rotary_query_shared,
            Q,
            row * Q0 + rope_copy_heads[:, None] * Q1 + 512 + rope_copy_d[None, :],
            mask=(rope_copy_heads[:, None] < H) | (H % 8 == 0),
        )
    if ASYNC_KV:
        copy_pos = split * BLOCK_S + gl.arange(
            0, BLOCK_S, gl.SliceLayout(1, copy_layout)
        )
        copy_in_bounds = (copy_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
        copy_slots = gl.load(S + slot_row * S0 + copy_pos, copy_in_bounds, -1)
        copy_active = copy_in_bounds & (copy_slots >= 0)
        copy_offsets = gl.where(copy_active, copy_slots, 0).to(gl.int32) * K0
        copy_d = gl.arange(0, 512, gl.SliceLayout(0, copy_layout))
        latent_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty, [BLOCK_S, 512], latent_layout
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            latent_shared,
            KV,
            copy_offsets[:, None] + copy_d[None, :],
            mask=copy_active[:, None],
        )
        if not ASYNC_QUERY:
            gl.amd.cdna4.async_copy.commit_group()
    elif BUFFER_KV:
        if DIRECT_OFFSETS:
            slot_offsets = slots.to(gl.int32) * K0
        else:
            slot_offsets = gl.where(active, slots, 0).to(gl.int32) * K0
        latent = gl.amd.cdna4.buffer_load(
            KV, slot_offsets[:, None] + d[None, :], active[:, None], 0.0
        )
    else:
        latent = gl.load(KV + slots[:, None] * K0 + d[None, :], active[:, None], 0.0)
    if ASYNC_QUERY:
        gl.amd.cdna4.async_copy.commit_group()
    else:
        q = gl.load(
            Q + query_row * Q0 + heads[:, None] * Q1 + query_d[None, :],
            (heads[:, None] < H) | (H % 8 == 0),
            0.0,
        )
    SCORE_ROWS: gl.constexpr = 8 if NATIVE_SCORE else 16
    rope_head_index = gl.arange(0, SCORE_ROWS, gl.SliceLayout(1, qk_a))
    rope_heads = group * 8 + (rope_head_index if NATIVE_SCORE else rope_head_index // 2)
    if WIDE_QUERY:
        rope_heads = rope_heads.to(gl.int64)
    rope_qd = gl.arange(0, 64, gl.SliceLayout(0, qk_a))
    if not ASYNC_ROPE:
        rotary_queries = gl.load(
            Q + query_row * Q0 + rope_heads[:, None] * Q1 + 512 + rope_qd[None, :],
            (rope_heads[:, None] < H) | (H % 8 == 0),
            0.0,
        )
    rope_pos = split * BLOCK_S + gl.arange(0, BLOCK_S, gl.SliceLayout(0, qk_b))
    rope_in_bounds = (rope_pos < SELECTED) | (SELECTED % BLOCK_S == 0)
    rope_slots = gl.load(S + slot_row * S0 + rope_pos, rope_in_bounds, -1).to(gl.int64)
    rope_active = rope_in_bounds & (rope_slots >= 0)
    rope_kd = gl.arange(0, 64, gl.SliceLayout(1, qk_b))
    if BUFFER_KV:
        if DIRECT_OFFSETS:
            rope_offsets = rope_slots.to(gl.int32) * K0
        else:
            rope_offsets = gl.where(rope_active, rope_slots, 0).to(gl.int32) * K0
        rotary_keys = gl.amd.cdna4.buffer_load(
            KV,
            rope_offsets[None, :] + 512 + rope_kd[:, None],
            rope_active[None, :],
            0.0,
        ).to(gl.bfloat16)
    else:
        rotary_keys = gl.load(
            KV + rope_slots[None, :] * K0 + 512 + rope_kd[:, None],
            rope_active[None, :],
            0.0,
        ).to(gl.bfloat16)
    if not ASYNC_QUERY:
        query_shared = gl.allocate_shared_memory(q.dtype, q.shape, query_layout, q)
    if ASYNC_KV:
        gl.amd.cdna4.async_copy.wait_group(0)
    else:
        latent_shared = gl.allocate_shared_memory(
            latent.dtype, [BLOCK_S, 512], latent_layout, latent
        )
    if ASYNC_ROPE:
        if NATIVE_SCORE:
            rotary_queries = rotary_query_shared.load(qk_a)
        else:
            rope_pair_rows = gl.arange(0, 16, gl.SliceLayout(1, qk_a)) // 2
            rope_query_indices = rope_pair_rows[:, None] + gl.full(
                (16, 64), 0, gl.int32, qk_a
            )
            rotary_queries = rotary_query_shared.gather(rope_query_indices, 0)
    if EARLY_PROB and (not RESERVE_PROB):
        probability_shared = gl.allocate_shared_memory(
            gl.bfloat16, [16, BLOCK_S], operand_shared
        )
    score = gl.full((SCORE_ROWS, BLOCK_S), 0, gl.float32, qk_matrix)
    for chunk in gl.static_range(0, 512 // QK_WIDTH):
        if NATIVE_SCORE:
            queries = query_shared.slice(chunk * QK_WIDTH, QK_WIDTH, 1).load(qk_a)
        else:
            pair_rows = gl.arange(0, 16, gl.SliceLayout(1, qk_a)) // 2
            query_indices = pair_rows[:, None] + gl.full(
                (16, QK_WIDTH), 0, gl.int32, qk_a
            )
            queries = query_shared.slice(chunk * QK_WIDTH, QK_WIDTH, 1).gather(
                query_indices, 0
            )
        keys = (
            latent_shared.slice(chunk * QK_WIDTH, QK_WIDTH, 1)
            .permute([1, 0])
            .load(qk_b)
            .to(gl.bfloat16)
        )
        score = gl.amd.cdna4.mfma(queries, keys, score)
    score = gl.amd.cdna4.mfma(rotary_queries, rotary_keys, score)
    if STAGE_SCORE:
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
    if NATIVE_SCORE:
        paired_probability = gl.reshape(
            gl.permute(gl.join(probability_hi, probability_lo), (0, 2, 1)),
            (16, BLOCK_S),
        )
    else:
        physical_head = gl.arange(0, 16, gl.SliceLayout(1, scores_layout))
        paired_probability = gl.where(
            physical_head[:, None] % 2 == 0, probability_hi, probability_lo
        )
    if EARLY_PROB or RESERVE_PROB:
        probability_shared.store(paired_probability)
    else:
        probability_shared = gl.allocate_shared_memory(
            gl.bfloat16, paired_probability.shape, operand_shared, paired_probability
        )
    if STREAM_VALUES:
        probabilities = probability_shared.load(pv_a)
    for fragment in gl.static_range(0, BLOCK_V // PV_FRAGMENT):
        if BLOCK_V == 128 and PRODUCER_WARPS == 2:
            if value_tile == 0:
                value_shared = latent_shared.slice(
                    fragment * PV_FRAGMENT, PV_FRAGMENT, 1
                )
            elif value_tile == 1:
                value_shared = latent_shared.slice(
                    128 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1
                )
            elif value_tile == 2:
                value_shared = latent_shared.slice(
                    256 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1
                )
            else:
                value_shared = latent_shared.slice(
                    384 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1
                )
            values = value_shared.load(pv_b).to(gl.bfloat16)
        elif BLOCK_V == 512:
            values = (
                latent_shared.slice(fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        elif BLOCK_V == 256:
            if value_tile == 0:
                values = (
                    latent_shared.slice(0 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                    .load(pv_b)
                    .to(gl.bfloat16)
                )
            else:
                values = (
                    latent_shared.slice(256 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                    .load(pv_b)
                    .to(gl.bfloat16)
                )
        elif value_tile == 0:
            values = (
                latent_shared.slice(0 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        elif value_tile == 1:
            values = (
                latent_shared.slice(128 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        elif value_tile == 2:
            values = (
                latent_shared.slice(256 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        else:
            values = (
                latent_shared.slice(384 + fragment * PV_FRAGMENT, PV_FRAGMENT, 1)
                .load(pv_b)
                .to(gl.bfloat16)
            )
        if not STREAM_VALUES:
            probabilities = probability_shared.load(pv_a)
        paired_numerator = gl.full((16, PV_FRAGMENT), 0, gl.float32, pv_matrix)
        paired_numerator = gl.amd.cdna4.mfma(probabilities, values, paired_numerator)
        numerator = gl.sum(gl.reshape(paired_numerator, (8, 2, PV_FRAGMENT)), 1)
        out_head = group * 8 + gl.arange(0, 8, gl.SliceLayout(1, numerator.type.layout))
        out_d = (
            value_tile * BLOCK_V
            + fragment * PV_FRAGMENT
            + gl.arange(0, PV_FRAGMENT, gl.SliceLayout(0, numerator.type.layout))
        )
        records = (partial_row * H + out_head) * SPLITS + split
        if BUFFER_PARTIAL:
            gl.amd.cdna4.buffer_store(
                ptr=Partial,
                offsets=records[:, None] * PARTIAL_STRIDE + out_d[None, :],
                stored_value=numerator,
                mask=(out_head[:, None] < H) | (H % 8 == 0),
            )
        else:
            gl.store(
                Partial + records[:, None] * PARTIAL_STRIDE + out_d[None, :],
                numerator,
                (out_head[:, None] < H) | (H % 8 == 0),
            )
    if NATIVE_SCORE:
        stat_head = group * 8 + gl.arange(0, 8, maximum.type.layout)
        stat_writer = ((stat_head < H) | (H % 8 == 0)) & (value_tile == 0)
    else:
        stat_head = group * 8 + physical_head // 2
        stat_writer = (
            ((stat_head < H) | (H % 8 == 0))
            & (physical_head % 2 == 0)
            & (value_tile == 0)
        )
    stat_record = (partial_row * H + stat_head) * SPLITS + split
    if COLOCATED:
        stat_offsets = stat_record * PARTIAL_STRIDE + 512
        gl.store(Partial + stat_offsets, maximum, stat_writer)
        gl.store(Partial + stat_offsets + 1, denominator, stat_writer)
        gl.store(
            Partial.to(gl.pointer_type(gl.int32)) + stat_offsets + 2,
            any_active.to(gl.int32),
            stat_writer,
        )
    else:
        gl.store(Stats + stat_record * 2, maximum, stat_writer)
        gl.store(Stats + stat_record * 2 + 1, denominator, stat_writer)
        gl.store(
            Valid + partial_row * SPLITS + split,
            any_active.to(gl.int32),
            (group == 0) & (value_tile == 0),
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
    WIDE_ADDRESS: gl.constexpr,
    PARTIAL_STRIDE: gl.constexpr,
    COLOCATED: gl.constexpr,
):
    row = gl.program_id(0)
    if WIDE_ADDRESS:
        row = row.to(gl.int64)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    wave_channels: gl.constexpr = BLOCK_D // MERGE_WARPS
    split_pack: gl.constexpr = 2 if BLOCK_D == 32 else 1
    layout: gl.constexpr = gl.BlockedLayout(
        [split_pack, 4],
        [256 // wave_channels, wave_channels // 4],
        [1, MERGE_WARPS],
        [1, 0],
    )
    s = gl.arange(0, REDUCTION_WIDTH, gl.SliceLayout(1, layout))
    d = tile * BLOCK_D + gl.arange(0, BLOCK_D, gl.SliceLayout(0, layout))
    record = (row * H + head) * SPLITS + s
    if COLOCATED:
        stat_offsets = record * PARTIAL_STRIDE + 512
        maximum = gl.load(Partial + stat_offsets, s < SPLITS, -float("inf"))
        denominator = gl.load(Partial + stat_offsets + 1, s < SPLITS, 0)
        count = gl.load(
            Partial.to(gl.pointer_type(gl.int32)) + stat_offsets + 2, s < SPLITS, 0
        )
    else:
        maximum = gl.load(Stats + record * 2, s < SPLITS, -float("inf"))
        denominator = gl.load(Stats + record * 2 + 1, s < SPLITS, 0)
        count = gl.load(Valid + row * SPLITS + s, s < SPLITS, 0)
    global_max = gl.maximum(-float("inf"), gl.max(maximum, 0))
    global_max = gl.where(gl.sum(count, 0) > 0, global_max, 0.0)
    factor = gl.where(
        count > 0, gl.exp2((maximum - global_max) * 1.4426950408889634), 0.0
    )
    numerators = gl.load(
        Partial + record[:, None] * PARTIAL_STRIDE + d[None, :],
        s[:, None] < SPLITS,
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
    assert m > 0 and h > 0 and (d == 576) and (query.dtype == torch.bfloat16)
    assert kv_cache.ndim == 2 and kv_cache.shape[1] == 576
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert selected_slots.ndim == 2 and selected_slots.shape[0] == m
    assert 0 < selected_slots.shape[1] <= 2048
    assert selected_slots.dtype in (torch.int32, torch.int64)
    assert query.stride(-1) == kv_cache.stride(-1) == selected_slots.stride(-1) == 1
    assert (
        query.device == kv_cache.device == selected_slots.device and softmax_scale > 0
    )
    block_size = 32 if m == 1 else 64 if m < 16 else 128
    value_width = 128 if m <= 2 else 256 if m <= 4 else 512
    splits = triton.cdiv(selected_slots.shape[1], block_size)
    colocated = m in (2, 8) or m >= 16
    partial_stride = 528 if colocated else 512
    cache_span = (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576
    buffer_kv = kv_cache.stride(0) >= 0 and cache_span < 2**31
    query_span = (m - 1) * query.stride(0) + (h - 1) * query.stride(1) + 576
    aligned_kv = (
        buffer_kv
        and kv_cache.stride(0) % 16 == 0
        and (kv_cache.storage_offset() % 16 == 0)
    )
    aligned_query = (
        query_span * 2 < 2**31
        and query.stride(0) >= 0
        and (query.stride(1) >= 0)
        and (query.stride(0) % 8 == 0)
        and (query.stride(1) % 8 == 0)
        and (query.storage_offset() % 8 == 0)
    )
    async_query = m in (1, 2, 4, 8) and aligned_kv and aligned_query
    async_kv = aligned_kv and (m >= 16 or async_query)
    async_rope = m in (8, 16) and aligned_query and async_kv
    producer_warps = 2 if m == 1 and async_query else 4
    qk_width = 512 if async_query else 128
    pair_slot = 4 if m == 2 and async_query else 16
    score_pack = (
        4 if m >= 16 or (m == 8 and async_query) else 2 if m in (1, 4, 8) else 4
    )
    latent_pad = 8 if m == 1 else 16
    stage_score = value_width != 256 and m != 8 and (m < 16)
    native_score = m in (1, 4, 8) or m >= 16
    query_warps = (
        1 if m == 2 or (m == 16 and async_rope) else 2 if m in (1, 4, 8, 16) else 4
    )
    early_prob = m in (2, 4)
    buffer_partial = (
        (m in (2, 4, 8) or m >= 16)
        and (not (m == 16 and async_rope))
        and (m * h * splits * partial_stride * 4 < 2**31)
    )
    slot_span = (m - 1) * selected_slots.stride(0) + selected_slots.shape[1]
    wide_query = query_span >= 2**31
    wide_slots = slot_span >= 2**31
    wide_partial = m * h * splits * partial_stride >= 2**31
    output = torch.empty((m, h, 512), device=query.device, dtype=torch.bfloat16)
    if m in (1, 4):
        partial_words = m * h * splits * partial_stride
        stats_words = m * h * splits * 2
        storage = torch.empty(
            partial_words + stats_words + m * splits,
            device=query.device,
            dtype=torch.float32,
        )
        partial = storage[:partial_words].view(m, h, splits, partial_stride)
        stats = storage[partial_words : partial_words + stats_words].view(
            m, h, splits, 2
        )
        valid = storage[partial_words + stats_words :].view(torch.int32).view(m, splits)
    else:
        partial = torch.empty(
            (m, h, splits, partial_stride), device=query.device, dtype=torch.float32
        )
        if colocated:
            stats = partial
            valid = partial
        else:
            stats = torch.empty(
                (m, h, splits, 2), device=query.device, dtype=torch.float32
            )
            valid = torch.empty((m, splits), device=query.device, dtype=torch.int32)
    shards = triton.cdiv(h, 8) * (512 // value_width)
    shard_first = m <= 4
    producer_grid = (m, shards, splits) if shard_first else (m, splits, shards)
    _split_attention[producer_grid](
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
        buffer_kv,
        stage_score,
        score_pack,
        latent_pad,
        shard_first,
        native_score,
        buffer_partial,
        query_warps,
        early_prob,
        wide_query,
        wide_slots,
        wide_partial,
        partial_stride,
        async_kv,
        colocated,
        async_query,
        qk_width,
        pair_slot,
        m == 8,
        512 if m == 8 and async_query else 256 if m == 8 else value_width,
        m == 8,
        producer_warps,
        async_rope,
        producer_warps == 2,
        num_warps=producer_warps,
        enable_fp_fusion=False,
        llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]
        if m >= 16 and async_kv
        else [],
    )
    if m <= 2:
        merge_width = 16 if m == 1 else 32
    else:
        merge_width = 64 if m <= 4 else 128 if m < 16 else 256
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
        wide_partial,
        partial_stride,
        colocated,
        num_warps=merge_warps,
        enable_fp_fusion=m <= 2,
    )
    return output
