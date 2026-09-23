# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m8.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _concat_matrix(left, right, AXIS: gl.constexpr, LAYOUT: gl.constexpr):
    paired = gl.join(left, right)
    if AXIS == 0:
        result = paired.permute(2, 0, 1).reshape(left.shape[0] * 2, left.shape[1])
    else:
        result = paired.permute(0, 2, 1).reshape(left.shape[0], left.shape[1] * 2)
    return gl.convert_layout(result, LAYOUT)


@gluon.jit
def _attention_partials(
    Q,
    KV,
    S,
    Part,
    Stats,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    H: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    SCALE,
    VALUE_TILE: gl.constexpr,
    QK_TILE: gl.constexpr,
    BUFFER: gl.constexpr,
    BUFFER_Q: gl.constexpr,
    SHARDS: gl.constexpr,
    PACK: gl.constexpr,
    MASK_PARTIALS: gl.constexpr,
    SPLIT_PADDING: gl.constexpr = 0,
    LDS_VEC: gl.constexpr = 16,
    STAGE_QUERY: gl.constexpr = False,
    RELOAD_VALID: gl.constexpr = False,
    STREAM_PARTIALS: gl.constexpr = False,
    PROB_MAX_PHASE: gl.constexpr = 8,
    STAGE_VALID: gl.constexpr = False,
    EARLY_PROB: gl.constexpr = False,
    EARLY_STATS: gl.constexpr = False,
    WAVE_SOFTMAX: gl.constexpr = False,
    JOIN_QUERY: gl.constexpr = False,
    FLAT_STORE_CACHE: gl.constexpr = "",
    STAGE_SCORE: gl.constexpr = False,
    RESERVE_PROB: gl.constexpr = False,
    PREFETCH_SHARD: gl.constexpr = False,
    INDEPENDENT_PV: gl.constexpr = False,
    JOINED_PV: gl.constexpr = False,
    PARTITION_MASK: gl.constexpr = False,
    PV_LOOKAHEAD: gl.constexpr = False,
    PACK_COLOR: gl.constexpr = 0,
    DIRECT_OUT: gl.constexpr = False,
):
    row = gl.program_id(0)
    if Q0 >= 2147483648 or S0 >= 2147483648:
        row = row.to(gl.int64)
    if H <= 16:
        head_block = 0
    else:
        head_block = gl.program_id(1) // SHARDS
    shard = gl.program_id(1) % SHARDS
    split = gl.program_id(2)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    rotary_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    matrix_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(0, matrix_layout, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, matrix_layout, 8)
    heads = head_block * 16 + gl.arange(0, 16, gl.SliceLayout(1, load_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, load_layout))
    kd = gl.arange(0, QK_TILE, gl.SliceLayout(0, load_layout))
    r = gl.arange(0, 64, gl.SliceLayout(0, rotary_layout))
    rotary_heads = head_block * 16 + gl.arange(0, 16, gl.SliceLayout(1, rotary_layout))
    if Q1 >= 2147483648:
        heads = heads.to(gl.int64)
        rotary_heads = rotary_heads.to(gl.int64)
    j = gl.arange(0, BLOCK, gl.SliceLayout(1, load_layout))
    pos = split * BLOCK + j
    if SELECTED % BLOCK == 0:
        slot = gl.load(S + row * S0 + pos).to(gl.int64)
    else:
        slot = gl.load(S + row * S0 + pos, pos < SELECTED, -1).to(gl.int64)
    if JOIN_QUERY:
        query_joined = gl.allocate_shared_memory(
            gl.bfloat16, (16, 1024), gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
        )
        if BUFFER_Q:
            q_latent = gl.amd.cdna4.buffer_load(
                Q + row * Q0, heads[:, None] * Q1 + d[None, :], heads[:, None] < H, 0.0
            )
            q_rotary = gl.amd.cdna4.buffer_load(
                Q + row * Q0,
                rotary_heads[:, None] * Q1 + 512 + r[None, :],
                rotary_heads[:, None] < H,
                0.0,
            )
        else:
            q_latent = gl.load(
                Q + row * Q0 + heads[:, None] * Q1 + d[None, :], heads[:, None] < H, 0.0
            )
            q_rotary = gl.load(
                Q + row * Q0 + rotary_heads[:, None] * Q1 + 512 + r[None, :],
                rotary_heads[:, None] < H,
                0.0,
            )
        query_joined.slice(0, 512, 1).store(q_latent)
        query_joined.slice(512, 64, 1).store(q_rotary)
    elif BUFFER_Q:
        qr = gl.amd.cdna4.buffer_load(
            Q + row * Q0,
            rotary_heads[:, None] * Q1 + 512 + r[None, :],
            rotary_heads[:, None] < H,
            0.0,
        )
    else:
        qr = gl.load(
            Q + row * Q0 + rotary_heads[:, None] * Q1 + 512 + r[None, :],
            rotary_heads[:, None] < H,
            0.0,
        )
    if SELECTED % BLOCK == 0:
        active = slot >= 0
    else:
        active = (pos < SELECTED) & (slot >= 0)
    rotary_active = gl.convert_layout(active, gl.SliceLayout(1, rotary_layout))
    if BUFFER:
        slot = slot.to(gl.int32)
        row_alignment: gl.constexpr = 16 if K0 == 0 else min(K0 & -K0, 16)
        kv_row = gl.multiple_of(gl.where(active, slot * K0, -2147483648), row_alignment)
        value = gl.amd.cdna4.buffer_load(KV, kv_row[:, None] + d[None, :])
        rotary_row = gl.convert_layout(kv_row, gl.SliceLayout(1, rotary_layout))
        rotary = gl.amd.cdna4.buffer_load(KV, rotary_row[:, None] + 512 + r[None, :])
    else:
        value = gl.load(KV + slot[:, None] * K0 + d[None, :], active[:, None], 0.0)
        rotary_slot = gl.convert_layout(slot, gl.SliceLayout(1, rotary_layout))
        rotary = gl.load(
            KV + rotary_slot[:, None] * K0 + 512 + r[None, :],
            rotary_active[:, None],
            0.0,
        )
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(LDS_VEC, 1, 16, [1, 0])
    value_shared = gl.allocate_shared_memory(
        KV.dtype.element_ty, (BLOCK, 512), shared_layout, value
    )
    if STAGE_VALID:
        validity_shared = gl.allocate_shared_memory(
            gl.int32,
            (BLOCK,),
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
            active.to(gl.int32),
        )
    if STAGE_SCORE:
        score_shared = gl.allocate_shared_memory(
            gl.float32, (16, BLOCK), gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
        )
    p_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, PROB_MAX_PHASE, [1, 0])
    if RESERVE_PROB:
        ph_shared = gl.allocate_shared_memory(gl.bfloat16, (16, BLOCK), p_layout)
        pl_shared = gl.allocate_shared_memory(gl.bfloat16, (16, BLOCK), p_layout)
    if STAGE_QUERY and (not JOIN_QUERY):
        if BUFFER_Q:
            q_full = gl.amd.cdna4.buffer_load(
                Q + row * Q0, heads[:, None] * Q1 + d[None, :], heads[:, None] < H, 0.0
            )
        else:
            q_full = gl.load(
                Q + row * Q0 + heads[:, None] * Q1 + d[None, :], heads[:, None] < H, 0.0
            )
        query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(
            8 if JOINED_PV else 16, 1, 8, [1, 0]
        )
        q_shared = gl.allocate_shared_memory(
            gl.bfloat16, (16, 512), query_shared_layout, q_full
        )
    if not JOIN_QUERY:
        qr_shared = gl.allocate_shared_memory(
            gl.bfloat16, (16, 64), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), qr
        )
    rotary_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    kr_shared = gl.allocate_shared_memory(
        KV.dtype.element_ty, (BLOCK, 64), rotary_shared_layout, rotary
    )
    dot = gl.zeros((16, BLOCK), gl.float32, matrix_layout)
    for key_start in gl.static_range(0, 512, QK_TILE):
        if JOIN_QUERY:
            q = query_joined.slice(key_start, QK_TILE, 1).load(a_layout)
        elif STAGE_QUERY:
            q = q_shared.slice(key_start, QK_TILE, 1).load(a_layout)
        elif BUFFER_Q:
            q = gl.amd.cdna4.buffer_load(
                Q + row * Q0,
                heads[:, None] * Q1 + key_start + kd[None, :],
                heads[:, None] < H,
                0.0,
            )
        else:
            q = gl.load(
                Q + row * Q0 + heads[:, None] * Q1 + key_start + kd[None, :],
                heads[:, None] < H,
                0.0,
            )
        key_operand = (
            value_shared.slice(key_start, QK_TILE, 1)
            .permute((1, 0))
            .load(b_layout)
            .to(gl.bfloat16)
        )
        dot = gl.amd.cdna4.mfma(gl.convert_layout(q, a_layout), key_operand, dot)
    if JOIN_QUERY:
        qr_operand = query_joined.slice(512, 64, 1).load(a_layout)
    else:
        qr_operand = qr_shared.load(a_layout)
    kr_operand = kr_shared.permute((1, 0)).load(b_layout).to(gl.bfloat16)
    dot = gl.amd.cdna4.mfma(qr_operand, kr_operand, dot)
    if PREFETCH_SHARD:
        if shard == 0:
            shard_values = value_shared.slice(0, 64, 1)
        elif shard == 1:
            shard_values = value_shared.slice(64, 64, 1)
        elif shard == 2:
            shard_values = value_shared.slice(128, 64, 1)
        elif shard == 3:
            shard_values = value_shared.slice(192, 64, 1)
        elif shard == 4:
            shard_values = value_shared.slice(256, 64, 1)
        elif shard == 5:
            shard_values = value_shared.slice(320, 64, 1)
        elif shard == 6:
            shard_values = value_shared.slice(384, 64, 1)
        else:
            shard_values = value_shared.slice(448, 64, 1)
        prefetched_value = shard_values.slice(0, VALUE_TILE, 1).load(b_layout)
    if WAVE_SOFTMAX:
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        if STAGE_SCORE:
            score_shared.store(dot)
            dot = score_shared.load(score_layout)
        else:
            dot = gl.convert_layout(dot, score_layout)
    else:
        score_layout: gl.constexpr = matrix_layout
    if STAGE_VALID:
        valid = validity_shared.load(gl.SliceLayout(0, score_layout)) != 0
    elif RELOAD_VALID:
        mask_pos = split * BLOCK + gl.arange(0, BLOCK, gl.SliceLayout(0, score_layout))
        if SELECTED % BLOCK == 0:
            mask_slot = gl.load(S + row * S0 + mask_pos)
            valid = mask_slot >= 0
        else:
            mask_slot = gl.load(S + row * S0 + mask_pos, mask_pos < SELECTED, -1)
            valid = (mask_pos < SELECTED) & (mask_slot >= 0)
    else:
        valid = gl.convert_layout(active, gl.SliceLayout(0, score_layout))
    if PARTITION_MASK:
        has_values = gl.max(valid.to(gl.int32), 0) != 0
    score = gl.where(valid[None, :], dot * SCALE, -float("inf"))
    maximum = gl.maximum(-float("inf"), gl.max(score, 1))
    probability = gl.where(
        valid[None, :], gl.exp2((score - maximum[:, None]) * 1.4426950408889634), 0.0
    )
    if not EARLY_PROB:
        denominator = gl.sum(probability, 1)
    p_hi = probability.to(gl.bfloat16)
    p_lo = (probability - p_hi.to(gl.float32)).to(gl.bfloat16)
    if RESERVE_PROB:
        ph_shared.store(p_hi)
        pl_shared.store(p_lo)
    else:
        ph_shared = gl.allocate_shared_memory(gl.bfloat16, (16, BLOCK), p_layout, p_hi)
        pl_shared = gl.allocate_shared_memory(gl.bfloat16, (16, BLOCK), p_layout, p_lo)
    if EARLY_PROB:
        denominator = gl.sum(probability, 1)
    if EARLY_STATS and (not DIRECT_OUT):
        stat_heads = head_block * 16 + gl.arange(0, 16, gl.SliceLayout(1, score_layout))
        if PACK:
            stat_index = (row * SPLITS + split) * H + stat_heads
        else:
            stat_index = (row * H + stat_heads) * SPLITS + split
        if shard == 0:
            if STREAM_PARTIALS:
                gl.amd.cdna4.buffer_store(
                    gl.join(maximum, denominator),
                    Stats,
                    gl.join(stat_index * 2, stat_index * 2 + 1),
                    gl.join(stat_heads < H, stat_heads < H),
                )
            else:
                gl.store(Stats + stat_index * 2, maximum, stat_heads < H)
                gl.store(Stats + stat_index * 2 + 1, denominator, stat_heads < H)
    if MASK_PARTIALS and (not PARTITION_MASK) or not EARLY_STATS or DIRECT_OUT:
        denominator = gl.convert_layout(denominator, gl.SliceLayout(1, matrix_layout))
    if not EARLY_STATS:
        maximum = gl.convert_layout(maximum, gl.SliceLayout(1, matrix_layout))
    p_hi = ph_shared.load(a_layout)
    p_lo = pl_shared.load(a_layout)
    if JOINED_PV:
        p_joined = _concat_matrix(p_hi, p_lo, 1, a_layout)
    oh = head_block * 16 + gl.arange(0, 16, gl.SliceLayout(1, matrix_layout))
    value_extent: gl.constexpr = 512 // SHARDS if PREFETCH_SHARD else 512
    if PV_LOOKAHEAD:
        next_value = value_shared.slice(0, VALUE_TILE, 1).load(b_layout)
    for tile_start in gl.static_range(0, value_extent, VALUE_TILE):
        if PREFETCH_SHARD or SHARDS == 1 or shard == tile_start // (512 // SHARDS):
            if PREFETCH_SHARD:
                value_start = shard * (512 // SHARDS) + tile_start
                pv_value = prefetched_value.to(gl.bfloat16)
            elif PV_LOOKAHEAD:
                value_start = tile_start
                pv_value = next_value.to(gl.bfloat16)
                if tile_start + VALUE_TILE < value_extent:
                    next_value = value_shared.slice(
                        tile_start + VALUE_TILE, VALUE_TILE, 1
                    ).load(b_layout)
            else:
                value_start = tile_start
                pv_value = (
                    value_shared.slice(tile_start, VALUE_TILE, 1)
                    .load(b_layout)
                    .to(gl.bfloat16)
                )
            if JOINED_PV:
                joined_value = _concat_matrix(pv_value, pv_value, 0, b_layout)
                numerator = gl.amd.cdna4.mfma(
                    p_joined,
                    joined_value,
                    gl.zeros((16, VALUE_TILE), gl.float32, matrix_layout),
                )
            else:
                numerator = gl.amd.cdna4.mfma(
                    gl.convert_layout(p_hi, a_layout),
                    pv_value,
                    gl.zeros((16, VALUE_TILE), gl.float32, matrix_layout),
                )
                if INDEPENDENT_PV:
                    residual_numerator = gl.amd.cdna4.mfma(
                        gl.convert_layout(p_lo, a_layout),
                        pv_value,
                        gl.zeros((16, VALUE_TILE), gl.float32, matrix_layout),
                    )
                    numerator = numerator + residual_numerator
                else:
                    numerator = gl.amd.cdna4.mfma(
                        gl.convert_layout(p_lo, a_layout), pv_value, numerator
                    )
            od = value_start + gl.arange(
                0, VALUE_TILE, gl.SliceLayout(0, matrix_layout)
            )
            if DIRECT_OUT:
                reciprocal = 1.0 / gl.where(denominator > 0, denominator, 1.0)
                result = numerator * reciprocal[:, None]
                gl.store(
                    Part + (row * H + oh[:, None]) * 512 + od[None, :],
                    result.to(gl.bfloat16),
                    oh[:, None] < H,
                )
            else:
                if PACK and PACK_COLOR:
                    physical_pack = od[None, :] // PACK ^ row * PACK_COLOR % (
                        512 // PACK
                    )
                    partial_offsets = (
                        (
                            (row * (512 // PACK) + physical_pack)
                            * (SPLITS + SPLIT_PADDING)
                            + split
                        )
                        * H
                        * PACK
                        + oh[:, None] * PACK
                        + od[None, :] % PACK
                    )
                elif PACK:
                    partial_offsets = (
                        (
                            (row * (512 // PACK) + od[None, :] // PACK)
                            * (SPLITS + SPLIT_PADDING)
                            + split
                        )
                        * H
                        * PACK
                        + oh[:, None] * PACK
                        + od[None, :] % PACK
                    )
                else:
                    partial_offsets = (
                        (row * H + oh[:, None]) * SPLITS * 512
                        + (od[None, :] // 64 * SPLITS + split) * 64
                        + od[None, :] % 64
                    )
                part_mask = oh[:, None] < H
                if PARTITION_MASK:
                    part_mask = part_mask & has_values
                elif MASK_PARTIALS:
                    part_mask = part_mask & (denominator[:, None] != 0)
                if STREAM_PARTIALS:
                    gl.amd.cdna4.buffer_store(
                        numerator, Part, partial_offsets, part_mask, cache=".cs"
                    )
                else:
                    gl.store(
                        Part + partial_offsets,
                        numerator,
                        part_mask,
                        cache_modifier=FLAT_STORE_CACHE,
                    )
    if not EARLY_STATS and (not DIRECT_OUT):
        index = (row * SPLITS + split) * H + oh
        gl.store(Stats + index * 2, maximum, oh < H)
        gl.store(Stats + index * 2 + 1, denominator, oh < H)


@gluon.jit
def _merge_channel_major(Part, Stats, O, SPLITS: gl.constexpr, BS: gl.constexpr):
    head = gl.program_id(1)
    tile = gl.program_id(0) * 2 + gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 1], [1, 0])
    s = gl.arange(0, BS, gl.SliceLayout(1, layout))
    d = tile * 32 + gl.arange(0, 32, gl.SliceLayout(0, layout))
    index = head * SPLITS + s
    maximum = gl.load(Stats + index * 2, s < SPLITS, -float("inf"))
    denominator = gl.load(Stats + index * 2 + 1, s < SPLITS, 0.0)
    global_max = gl.max(maximum, 0)
    safe_max = gl.where(global_max == -float("inf"), 0.0, global_max)
    weight = gl.exp2((maximum - safe_max) * 1.4426950408889634)
    offsets = (
        head * SPLITS * 512
        + (d[None, :] // 64 * SPLITS + s[:, None]) * 64
        + d[None, :] % 64
    )
    numerator = gl.load(Part + offsets, (s < SPLITS)[:, None], 0.0)
    total = gl.sum(denominator * weight, 0)
    reciprocal = 1.0 / gl.where(total > 0, total, 1.0)
    result = gl.sum(numerator * weight[:, None], 0) * reciprocal
    gl.store(O + head * 512 + d, result.to(gl.bfloat16))


@gluon.jit
def _merge_packed(
    Part,
    Stats,
    O,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    BS: gl.constexpr,
    PACK: gl.constexpr,
    HEAD_TILE: gl.constexpr,
    SPLIT_PADDING: gl.constexpr = 0,
    MERGE_PACKS: gl.constexpr = 1,
    MERGE_WARPS: gl.constexpr = 1,
    BUFFER_PART: gl.constexpr = False,
    PAIRED_STATS: gl.constexpr = False,
    PART_CACHE: gl.constexpr = "",
    SPLIT_LOCAL: gl.constexpr = False,
    ELIDE_HEAD_MASK: gl.constexpr = False,
    BOUNDED_PART: gl.constexpr = False,
    BUFFER_OUT: gl.constexpr = False,
    NATIVE_EXP: gl.constexpr = False,
    MERGE_SHARDS: gl.constexpr = 1,
    PACK_COLOR: gl.constexpr = 0,
):
    row = gl.program_id(0)
    if MERGE_SHARDS > 1:
        HEAD_BLOCKS: gl.constexpr = gl.cdiv(H, HEAD_TILE)
        TILES_PER_SHARD: gl.constexpr = 512 // PACK // MERGE_PACKS // MERGE_SHARDS
        head_block = gl.program_id(2) % HEAD_BLOCKS
        tile = gl.program_id(1) * TILES_PER_SHARD + gl.program_id(2) // HEAD_BLOCKS
    else:
        head_block = gl.program_id(1)
        tile = gl.program_id(2)
    lane_order: gl.constexpr = (2, 0, 1) if SPLIT_LOCAL else (2, 1, 0)
    lanes_d: gl.constexpr = min(PACK // 4, 16)
    lanes_h: gl.constexpr = min(HEAD_TILE, 64 // lanes_d)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4],
        [64 // lanes_d // lanes_h, lanes_h, lanes_d],
        [1, 1, MERGE_WARPS],
        lane_order,
    )
    sh_layout: gl.constexpr = gl.SliceLayout(2, layout)
    s = gl.arange(0, BS, gl.SliceLayout(1, sh_layout))
    h = head_block * HEAD_TILE + gl.arange(0, HEAD_TILE, gl.SliceLayout(0, sh_layout))
    d = gl.arange(0, PACK * MERGE_PACKS, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    index = (row * SPLITS + s[:, None]) * H + h[None, :]
    if ELIDE_HEAD_MASK and H % HEAD_TILE == 0:
        head_valid = gl.full((HEAD_TILE,), True, gl.int1, gl.SliceLayout(0, sh_layout))
    else:
        head_valid = h < H
    active = (s[:, None] < SPLITS) & head_valid[None, :]
    if BUFFER_PART and PAIRED_STATS:
        stats_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 2],
            [64 // lanes_d // lanes_h, lanes_h, lanes_d],
            [1, 1, MERGE_WARPS],
            lane_order,
        )
        pair = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, stats_layout)))
        stat_index = gl.convert_layout(index, gl.SliceLayout(2, stats_layout))
        stat_mask = gl.convert_layout(active, gl.SliceLayout(2, stats_layout))
        records = gl.amd.cdna4.buffer_load(
            Stats,
            stat_index[:, :, None] * 2 + pair[None, None, :],
            stat_mask[:, :, None],
            0.0,
        )
        max_record, denom_record = gl.split(records)
        maximum = gl.convert_layout(max_record, sh_layout)
        denominator = gl.convert_layout(denom_record, sh_layout)
        maximum = gl.where(active, maximum, -float("inf"))
    elif BUFFER_PART:
        maximum = gl.amd.cdna4.buffer_load(Stats, index * 2, active, -float("inf"))
        denominator = gl.amd.cdna4.buffer_load(Stats, index * 2 + 1, active, 0.0)
    else:
        maximum = gl.load(Stats + index * 2, active, -float("inf"))
        denominator = gl.load(Stats + index * 2 + 1, active, 0.0)
    global_max = gl.max(maximum, 0)
    safe_max = gl.where(global_max == -float("inf"), 0.0, global_max)
    if NATIVE_EXP:
        weight = gl.exp2((maximum - safe_max[None, :]) * 1.4426950408889634)
    else:
        weight = gl.exp(maximum - safe_max[None, :])
    offsets = (
        (
            (row * (512 // PACK) + tile * MERGE_PACKS + d[None, None, :] // PACK)
            * (SPLITS + SPLIT_PADDING)
            + s[:, None, None]
        )
        * H
        * PACK
        + h[None, :, None] * PACK
        + d[None, None, :] % PACK
    )
    if PACK_COLOR:
        physical_tile = tile * MERGE_PACKS + d[
            None, None, :
        ] // PACK ^ row * PACK_COLOR % (512 // PACK)
        offsets = (
            (
                (row * (512 // PACK) + physical_tile) * (SPLITS + SPLIT_PADDING)
                + s[:, None, None]
            )
            * H
            * PACK
            + h[None, :, None] * PACK
            + d[None, None, :] % PACK
        )
    part_mask = (active & (denominator != 0))[:, :, None]
    if BUFFER_PART and BOUNDED_PART and (MERGE_PACKS == 1):
        record = (
            (row * (512 // PACK) + tile) * (SPLITS + SPLIT_PADDING) + s[:, None]
        ) * H * PACK + h[None, :] * PACK
        if PACK_COLOR:
            record = (
                (row * (512 // PACK) + (tile ^ row * PACK_COLOR % (512 // PACK)))
                * (SPLITS + SPLIT_PADDING)
                + s[:, None]
            ) * H * PACK + h[None, :] * PACK
        record = gl.multiple_of(
            gl.where(active & (denominator != 0), record, 1 << 29), [PACK, PACK]
        )
        values = gl.amd.cdna4.buffer_load(
            Part, record[:, :, None] + d[None, None, :], cache=PART_CACHE
        )
    elif BUFFER_PART:
        values = gl.amd.cdna4.buffer_load(
            Part, offsets, part_mask, 0.0, cache=PART_CACHE
        )
    else:
        values = gl.load(Part + offsets, part_mask, 0.0, cache_modifier=PART_CACHE)
    total = gl.sum(denominator * weight, 0)
    numerator = gl.sum(values * weight[:, :, None], 0)
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    total = gl.convert_layout(total, gl.SliceLayout(1, out_layout))
    reciprocal = 1.0 / gl.where(total > 0, total, 1.0)
    out = numerator * reciprocal[:, None]
    oh = head_block * HEAD_TILE + gl.arange(0, HEAD_TILE, gl.SliceLayout(1, out_layout))
    od = tile * PACK * MERGE_PACKS + gl.arange(
        0, PACK * MERGE_PACKS, gl.SliceLayout(0, out_layout)
    )
    if BUFFER_OUT and H % HEAD_TILE == 0:
        gl.amd.cdna4.buffer_store(
            out.to(gl.bfloat16), O, (row * H + oh[:, None]) * 512 + od[None, :]
        )
    elif ELIDE_HEAD_MASK and H % HEAD_TILE == 0:
        gl.store(O + (row * H + oh[:, None]) * 512 + od[None, :], out.to(gl.bfloat16))
    else:
        gl.store(
            O + (row * H + oh[:, None]) * 512 + od[None, :],
            out.to(gl.bfloat16),
            oh[:, None] < H,
        )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, h, _ = query.shape
    selected = selected_slots.shape[1]
    block = 64 if m < 16 else 128
    splits = triton.cdiv(selected, block)
    shards = 8 if m == 1 else 4 if m == 2 else 2 if m == 4 else 1
    pack = 0 if m == 1 else 8
    value_tile = 64 if m == 1 or (2 < m < 8 and m != 4) else 128
    qk_tile = 512 if m >= 16 else 256
    head_tile = 4 if m == 2 else 8 if m < 16 else 16
    mask_partials = m in (2, 4) or m >= 8
    pad = 8 if m == 2 else 0
    lds_vec = 8 if m >= 16 else 16
    merge_packs = 4 if 8 <= m < 16 else 1
    merge_warps = merge_packs
    staged = m in (1, 2, 4, 8)
    buffer_limit = 2**31 - 2
    buffer_k = (
        0 <= kv_cache.stride(0) <= buffer_limit
        and (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576 <= buffer_limit
    )
    buffer_q = (
        0 <= query.stride(1) * 2 <= buffer_limit
        and ((h - 1) * query.stride(1) + 576) * 2 <= buffer_limit
    )
    output = torch.empty((m, h, 512), device=query.device, dtype=torch.bfloat16)
    if selected <= 128:
        short_block = max(32, triton.next_power_of_2(selected))
        _attention_partials[m, triton.cdiv(h, 16) * 8, 1](
            query,
            kv_cache,
            selected_slots,
            output,
            output,
            query.stride(0),
            query.stride(1),
            kv_cache.stride(0),
            selected_slots.stride(0),
            h,
            selected,
            1,
            short_block,
            softmax_scale,
            64,
            256,
            buffer_k,
            buffer_q,
            8,
            0,
            False,
            STAGE_QUERY=True,
            STAGE_VALID=True,
            EARLY_PROB=True,
            EARLY_STATS=True,
            WAVE_SOFTMAX=True,
            RESERVE_PROB=True,
            PREFETCH_SHARD=True,
            PROB_MAX_PHASE=16,
            DIRECT_OUT=True,
            num_warps=4,
            enable_fp_fusion=False,
            allow_flush_denorm=True,
        )
        return output
    partial_shape = (
        (m, 512 // pack, splits + pad, h, pack) if pack else (m, h, 8, splits, 64)
    )
    stats_shape = (m, splits, h, 2) if pack else (m, h, splits, 2)
    if m in (1, 8, 16):
        partial_words = m * h * splits * 512
        gap_words = 1024 if m == 16 else 0
        stats_start = partial_words + gap_words
        storage = torch.empty(
            stats_start + m * h * splits * 2, device=query.device, dtype=torch.float32
        )
        partial = storage[:partial_words].view(partial_shape)
        stats = storage[stats_start:].view(stats_shape)
    else:
        partial = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
        stats = torch.empty(stats_shape, device=query.device, dtype=torch.float32)
    buffer_part = max(partial.numel(), stats.numel()) * 4 <= buffer_limit
    _attention_partials[m, triton.cdiv(h, 16) * shards, splits](
        query,
        kv_cache,
        selected_slots,
        partial,
        stats,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        h,
        selected,
        splits,
        block,
        softmax_scale,
        value_tile,
        qk_tile,
        buffer_k,
        buffer_q,
        shards,
        pack,
        mask_partials,
        pad,
        lds_vec,
        STAGE_QUERY=staged or m >= 16,
        RELOAD_VALID=m >= 16,
        STREAM_PARTIALS=m == 8 and buffer_part,
        PROB_MAX_PHASE=16 if m == 1 else 8,
        STAGE_VALID=staged,
        EARLY_PROB=staged or m >= 16,
        EARLY_STATS=staged,
        WAVE_SOFTMAX=staged,
        JOIN_QUERY=m >= 16,
        FLAT_STORE_CACHE=".cs" if m >= 16 else "",
        STAGE_SCORE=m in (2, 4, 8),
        RESERVE_PROB=m == 1,
        PREFETCH_SHARD=m == 1,
        INDEPENDENT_PV=m >= 16,
        JOINED_PV=m == 8,
        PARTITION_MASK=m in (2, 4),
        PV_LOOKAHEAD=m == 16,
        PACK_COLOR=4 if m == 16 else 0,
        num_warps=4,
        enable_fp_fusion=False,
        allow_flush_denorm=True,
    )
    if pack:
        merge_shards = shards if m in (2, 4) else 1
        head_blocks = triton.cdiv(h, head_tile)
        merge_tiles = 512 // pack // merge_packs
        merge_grid = (
            (m, merge_shards, head_blocks * (merge_tiles // merge_shards))
            if merge_shards > 1
            else (m, head_blocks, merge_tiles)
        )
        _merge_packed[merge_grid](
            partial,
            stats,
            output,
            h,
            splits,
            triton.next_power_of_2(max(1, splits)),
            pack,
            head_tile,
            pad,
            merge_packs,
            merge_warps,
            buffer_part,
            m == 2 or 8 <= m < 16,
            PART_CACHE=".cg" if m == 8 or m >= 16 else "",
            SPLIT_LOCAL=m == 2,
            ELIDE_HEAD_MASK=m in (2, 4, 8),
            BOUNDED_PART=m in (2, 16),
            BUFFER_OUT=m in (4, 8, 16) and output.numel() * 2 <= buffer_limit,
            NATIVE_EXP=m == 8,
            MERGE_SHARDS=merge_shards,
            PACK_COLOR=4 if m == 16 else 0,
            num_warps=merge_warps,
            enable_fp_fusion=m in (4, 16),
        )
    else:
        _merge_channel_major[8, m * h, 2](
            partial,
            stats,
            output,
            splits,
            triton.next_power_of_2(max(1, splits)),
            num_warps=1,
            enable_fp_fusion=False,
            allow_flush_denorm=True,
        )
    return output
