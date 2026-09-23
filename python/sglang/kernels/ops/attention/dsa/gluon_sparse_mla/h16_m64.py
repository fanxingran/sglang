# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m64.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_PARTIAL_PITCH = gl.constexpr(520)


@gluon.jit
def _natural_exp(x):
    return gl.exp2(x * 1.4426950408889634)


@gluon.jit
def _or_activity(a, b):
    return a | b


@gluon.jit
def _load_slots(S, row, positions, S0: gl.constexpr, SELECTED: gl.constexpr):
    mask = (SELECTED % 128 == 0) | (positions < SELECTED)
    return gl.load(S + row * S0 + positions, mask, -1)


@gluon.jit
def _publish_m64_slots(
    S, row, split, S0: gl.constexpr, SELECTED: gl.constexpr, K0: gl.constexpr
):
    slots_layout: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [1, 4], [0, 1])
    tokens = gl.arange(0, 64, gl.SliceLayout(1, slots_layout))
    fields = gl.arange(0, 4, gl.SliceLayout(0, slots_layout))
    positions = split * 192 + tokens[:, None] + fields[None, :] * 64
    raw = gl.amd.cdna4.buffer_load(
        S + row * S0, positions, (positions < SELECTED) & (fields[None, :] < 3), -1
    )
    offsets = gl.where(raw >= 0, raw.to(gl.int32) * (K0 // 16), -(1 << 27))
    packed = offsets.reshape((256,))
    shared = gl.allocate_shared_memory(
        gl.int32, [256], gl.SwizzledSharedLayout(1, 1, 1, [0]), value=packed
    )
    ballot = gl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $1",
        "=s,v",
        [(raw >= 0).to(gl.int32)],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    first = gl.gather(ballot, gl.full((1, 4), 0, gl.int32, slots_layout), 0)
    field_activity = (gl.sum(first, 0) != 0).to(gl.int32)
    activity = gl.reduce(field_activity << fields, 0, _or_activity)
    return (shared, activity)


@gluon.jit
def _allocate_m64_kv(DTYPE: gl.constexpr):
    key_shared = gl.allocate_shared_memory(
        DTYPE,
        [64, 512],
        gl.PaddedSharedLayout(
            [[1024, 8]],
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
                [32, 0],
            ],
            [],
            [64, 512],
        ),
    )
    rotary_shared = gl.allocate_shared_memory(
        DTYPE,
        [64, 64],
        gl.PaddedSharedLayout(
            [[1024, 8]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [1, 0],
                [2, 0],
            ],
            [],
            [64, 64],
        ),
    )
    return (key_shared, rotary_shared)


@gluon.jit
def _prefetch_packed_m64(
    KV,
    key_shared,
    rotary_shared,
    slots_shared,
    dd,
    rr,
    TILE: gl.constexpr,
    LOAD_LAYOUT: gl.constexpr,
    ROTARY_LAYOUT: gl.constexpr,
):
    nn = gl.arange(0, 64, gl.SliceLayout(1, LOAD_LAYOUT))
    rn = gl.arange(0, 64, gl.SliceLayout(1, ROTARY_LAYOUT))
    offset = slots_shared.gather(nn * 4 + TILE, 0)
    rotary_offset = slots_shared.gather(rn * 4 + TILE, 0)
    _copy_kv(KV, key_shared, rotary_shared, offset, rotary_offset, dd, rr)


@gluon.jit
def _copy_kv(KV, key_shared, rotary_shared, offset, rotary_offset, dd, rr):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        rotary_shared,
        KV,
        rotary_offset[:, None] * 16 + 512 + rr[None, :],
        cache_modifier=".cg",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        key_shared, KV, offset[:, None] * 16 + dd[None, :], cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _copy_m32_kv(KV, key_shared, rotary_shared, offset, rotary_offset, dd, rr):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        key_shared, KV, offset[:, None] * 16 + dd[None, :], cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        rotary_shared,
        KV,
        rotary_offset[:, None] * 16 + 512 + rr[None, :],
        cache_modifier=".cg",
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _store_partial_tags_and_wide(
    Wide,
    RecordFlag,
    record,
    features,
    numerator,
    denominator,
    supplied_peak,
    STAT_PITCH: gl.constexpr = 1,
):
    striped = (
        numerator.reshape((16, 8, 4, 16)).permute((0, 2, 1, 3)).reshape((16, 4, 128))
    )
    peak = gl.convert_layout(
        supplied_peak, gl.SliceLayout(2, striped.type.layout), assert_trivial=True
    )
    flag_layout: gl.constexpr = peak.type.layout
    local_den = gl.convert_layout(denominator, gl.SliceLayout(1, flag_layout))
    wide = peak > local_den[:, None] * 0.5
    local_record = gl.convert_layout(record, gl.SliceLayout(1, flag_layout))
    group = gl.arange(0, 4, gl.SliceLayout(0, flag_layout))
    tag_bytes = RecordFlag.to(gl.pointer_type(gl.int8))
    gl.store(
        tag_bytes + local_record[:, None] * (4 * STAT_PITCH) + group[None, :],
        wide.to(gl.int8),
    )
    wide_features = gl.broadcast(wide[:, :, None], striped)[0]
    wide_features = (
        wide_features.reshape((16, 4, 8, 16)).permute((0, 2, 1, 3)).reshape((16, 512))
    )
    wide_features = gl.convert_layout(wide_features, numerator.type.layout)
    wide_features = gl.max_constancy(wide_features, [1, 16])
    gl.store(
        Wide + record[:, None] * _PARTIAL_PITCH + features[None, :],
        numerator,
        wide_features,
    )


@gluon.jit
def _load_query_tiles(Q, row, Q0: gl.constexpr, Q1: gl.constexpr):
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    hh = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    dd = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    rr = gl.arange(0, 64, gl.SliceLayout(0, query_layout))
    qq = gl.load(Q + row * Q0 + hh[:, None] * Q1 + dd[None, :])
    qr = gl.load(Q + row * Q0 + hh[:, None] * Q1 + 512 + rr[None, :])
    return (qq, qr)


@gluon.jit
def _stage_matrix(value, PADDING: gl.constexpr):
    return gl.allocate_shared_memory(
        value.dtype,
        value.shape,
        gl.PaddedSharedLayout.with_identity_for(
            [[value.shape[1], PADDING]], value.shape, [1, 0]
        ),
        value=value,
    )


@gluon.jit
def _stage_query_halves(qq, qr, dot_a: gl.constexpr):
    q0, q1 = qq.reshape((16, 2, 256)).permute((0, 2, 1)).split()
    q0_shared = _stage_matrix(q0, 16)
    q1_shared = _stage_matrix(q1, 16)
    qr_shared = _stage_matrix(qr, 16)
    return (q0_shared.load(dot_a), q1_shared.load(dot_a), qr_shared.load(dot_a))


@gluon.jit
def _wave_maximum(score, shared):
    grouped = score.reshape((16, score.shape[1] // 64, 4, 16))
    partial = gl.max(gl.max(grouped, 1), 2)
    shared.store(partial)
    gl.barrier()
    read_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [4, 1], [0, 1])
    parts = gl.amd.cdna4.async_copy.load_shared_relaxed(shared, read_layout)
    return gl.convert_layout(gl.max(parts, 1), gl.SliceLayout(1, score.type.layout))


@gluon.jit
def _store_statistics(Denominator, record, denominator, maximum):
    values = gl.join(denominator, maximum)
    pair_layout: gl.constexpr = values.type.layout
    rr = gl.convert_layout(record, gl.SliceLayout(1, pair_layout))
    ff = gl.arange(0, 2, gl.SliceLayout(0, pair_layout))
    gl.store(Denominator + rr[:, None] * 4 + ff[None, :], values)


@gluon.jit
def _split_attention_m64(
    Q,
    KV,
    S,
    Workspace,
    scale,
    ROWS: gl.constexpr,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
):
    PAYLOAD: gl.constexpr = ROWS * SPLITS * 16 * _PARTIAL_PITCH
    Numerator = Workspace.to(gl.pointer_type(gl.float16))
    WideNumerator = Workspace + PAYLOAD // 2
    RecordFlag = (Workspace + 3 * PAYLOAD // 2).to(gl.pointer_type(gl.int32))
    Denominator = Workspace + 3 * PAYLOAD // 2 + 2
    row = gl.program_id(0)
    split = gl.program_id(1)
    BLOCK_H: gl.constexpr = 16
    load_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [32, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[64, 512],
    )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    dd = gl.arange(0, 512, gl.SliceLayout(0, load_layout))
    rotary_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [32, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[64, 64],
    )
    rr = gl.arange(0, 64, gl.SliceLayout(0, rotary_layout))
    query_tile, rotary_query_tile = _load_query_tiles(Q, row, Q0, Q1)
    nn = gl.arange(0, 64, gl.SliceLayout(1, load_layout))
    nh = gl.arange(0, BLOCK_H, gl.SliceLayout(1, mma_layout))
    nv = gl.arange(0, 512, gl.SliceLayout(0, mma_layout))
    slots_shared, activity = _publish_m64_slots(S, row, split, S0, SELECTED, K0)
    any_tokens = activity != 0
    if any_tokens:
        max_score = gl.full(
            (BLOCK_H,), -float("inf"), gl.float32, gl.SliceLayout(1, mma_layout)
        )
        numerator = gl.full((BLOCK_H, 512), 0, gl.float32, mma_layout)
        key_shared, rotary_shared = _allocate_m64_kv(KV.dtype.element_ty)
        _prefetch_packed_m64(
            KV,
            key_shared,
            rotary_shared,
            slots_shared,
            dd,
            rr,
            0,
            load_layout,
            rotary_layout,
        )
        q0, q1, qr = _stage_query_halves(query_tile, rotary_query_tile, dot_a)
        max_shared = gl.allocate_shared_memory(
            gl.float32, [4, 16, 4], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        den_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
        for tile in gl.static_range(3):
            score_nn = gl.arange(0, 64, gl.SliceLayout(0, mma_layout))
            score_offset = slots_shared.gather(score_nn * 4 + tile, 0)
            active_score = score_offset >= 0
            if tile < 2:
                rotary_nn = gl.arange(0, 64, gl.SliceLayout(1, rotary_layout))
                next_offset = slots_shared.gather(nn * 4 + tile + 1, 0)
                next_rotary_offset = slots_shared.gather(rotary_nn * 4 + tile + 1, 0)
            gl.amd.cdna4.async_copy.wait_group(0)
            key_t = gl.amd.cdna4.async_copy.load_shared_relaxed(
                key_shared.permute((1, 0)), dot_b
            )
            score = gl.full((BLOCK_H, 64), 0, gl.float32, mma_layout)
            k0, k1 = key_t.reshape((2, 256, 64)).permute((1, 2, 0)).split()
            k0 = gl.convert_layout(k0, dot_b).to(gl.bfloat16)
            score0 = gl.amd.cdna4.mfma(q0, k0, score)
            k1 = gl.convert_layout(k1, dot_b).to(gl.bfloat16)
            score1 = gl.amd.cdna4.mfma(q1, k1, score)
            rotary_t = gl.amd.cdna4.async_copy.load_shared_relaxed(
                rotary_shared.permute((1, 0)), dot_b
            ).to(gl.bfloat16)
            score1 = gl.amd.cdna4.mfma(qr, rotary_t, score1)
            score = (score0 + score1) * scale
            score = gl.where(active_score[None, :], score, -float("inf"))
            next_max = gl.maximum(
                max_score, _wave_maximum(score, max_shared.index(tile))
            )
            seen_active = activity & (1 << tile + 1) - 1 != 0
            safe_max = gl.where(seen_active, next_max, 0.0)
            alpha = _natural_exp(max_score - safe_max)
            probability = gl.where(
                active_score[None, :], _natural_exp(score - safe_max[:, None]), 0.0
            )
            local_denominator = gl.sum(probability.reshape((16, 4, 16)), 2)
            denominator_shared = gl.allocate_shared_memory(
                gl.float32,
                [16, 4],
                gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
                value=local_denominator,
            )
            numerator *= alpha[:, None]
            p_hi = probability.to(gl.bfloat16)
            p_lo = (probability - p_hi.to(gl.float32)).to(gl.bfloat16)
            p_hi_shared = _stage_matrix(p_hi, 8)
            p_lo_shared = _stage_matrix(p_lo, 8)
            value = gl.amd.cdna4.async_copy.load_shared_relaxed(key_shared, dot_b)
            gl.barrier()
            if tile < 2:
                _copy_kv(
                    KV,
                    key_shared,
                    rotary_shared,
                    next_offset,
                    next_rotary_offset,
                    dd,
                    rr,
                )
            p_lo = gl.amd.cdna4.async_copy.load_shared_relaxed(p_lo_shared, dot_a)
            den_parts = gl.amd.cdna4.async_copy.load_shared_relaxed(
                denominator_shared, den_layout
            )
            tile_denominator = gl.convert_layout(
                gl.sum(den_parts, 1), gl.SliceLayout(1, mma_layout)
            )
            if tile == 0:
                denominator = tile_denominator
            else:
                denominator = denominator * alpha + tile_denominator
            p_hi = p_hi_shared.load(dot_a)
            if tile < 2:
                numerator = _tile_values_grouped(
                    p_lo, p_hi, value, numerator, dot_b, mma_layout
                )
            else:
                record = (row * SPLITS + split) * 16 + nh
                numerator, peak = _last_tile_values(
                    p_lo, p_hi, value, numerator, Numerator, record, dot_b, mma_layout
                )
                _store_partial_tags_and_wide(
                    WideNumerator,
                    RecordFlag,
                    record,
                    nv,
                    numerator,
                    denominator,
                    peak,
                    STAT_PITCH=4,
                )
            max_score = next_max
        record = (row * SPLITS + split) * 16 + nh
        _store_statistics(Denominator, record, denominator, max_score)
    else:
        record = (row * SPLITS + split) * 16 + nh
        gl.store(RecordFlag + record * 4, -1)
        empty_den = gl.full((16,), 0.0, gl.float32, record.type.layout)
        empty_max = gl.full((16,), -float("inf"), gl.float32, record.type.layout)
        _store_statistics(Denominator, record, empty_den, empty_max)


@gluon.jit
def _tile_score(
    qq, qr, key_shared, rotary_shared, dot_b: gl.constexpr, mma: gl.constexpr
):
    key = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_shared.permute((1, 0)), dot_b
    ).to(gl.bfloat16)
    rotary = gl.amd.cdna4.async_copy.load_shared_relaxed(
        rotary_shared.permute((1, 0)), dot_b
    ).to(gl.bfloat16)
    score = gl.full((16, 64), 0.0, gl.float32, mma)
    score = gl.amd.cdna4.mfma(qq, key, score)
    return gl.amd.cdna4.mfma(qr, rotary, score)


@gluon.jit
def _accumulate_value_group(
    p_lo, p_hi, value, numerator, dot_b: gl.constexpr, mma: gl.constexpr
):
    value = gl.convert_layout(value, dot_b).to(gl.bfloat16)
    numerator = gl.convert_layout(numerator, mma)
    numerator = gl.amd.cdna4.mfma(p_lo, value, numerator)
    return gl.amd.cdna4.mfma(p_hi, value, numerator)


@gluon.jit
def _split_feature_eighths(value):
    rows: gl.constexpr = value.shape[0]
    left, right = value.reshape((rows, 2, 256)).permute((0, 2, 1)).split()
    quarter0, quarter1 = left.reshape((rows, 2, 128)).permute((0, 2, 1)).split()
    quarter2, quarter3 = right.reshape((rows, 2, 128)).permute((0, 2, 1)).split()
    v0, v1 = quarter0.reshape((rows, 2, 64)).permute((0, 2, 1)).split()
    v2, v3 = quarter1.reshape((rows, 2, 64)).permute((0, 2, 1)).split()
    v4, v5 = quarter2.reshape((rows, 2, 64)).permute((0, 2, 1)).split()
    v6, v7 = quarter3.reshape((rows, 2, 64)).permute((0, 2, 1)).split()
    return (v0, v1, v2, v3, v4, v5, v6, v7)


@gluon.jit
def _join_feature_eighths(n0, n1, n2, n3, n4, n5, n6, n7):
    quarter0 = gl.join(n0, n1).permute((0, 2, 1)).reshape((16, 128))
    quarter1 = gl.join(n2, n3).permute((0, 2, 1)).reshape((16, 128))
    quarter2 = gl.join(n4, n5).permute((0, 2, 1)).reshape((16, 128))
    quarter3 = gl.join(n6, n7).permute((0, 2, 1)).reshape((16, 128))
    left = gl.join(quarter0, quarter1).permute((0, 2, 1)).reshape((16, 256))
    right = gl.join(quarter2, quarter3).permute((0, 2, 1)).reshape((16, 256))
    return gl.join(left, right).permute((0, 2, 1)).reshape((16, 512))


@gluon.jit
def _tile_values_grouped(
    p_lo, p_hi, value, numerator, dot_b: gl.constexpr, mma: gl.constexpr
):
    v0, v1, v2, v3, v4, v5, v6, v7 = _split_feature_eighths(value)
    n0, n1, n2, n3, n4, n5, n6, n7 = _split_feature_eighths(numerator)
    v0 = gl.convert_layout(v0, dot_b).to(gl.bfloat16)
    n0 = gl.convert_layout(n0, mma)
    n0 = gl.amd.cdna4.mfma(p_lo, v0, n0)
    v1 = gl.convert_layout(v1, dot_b).to(gl.bfloat16)
    n0 = gl.amd.cdna4.mfma(p_hi, v0, n0)
    n1 = gl.convert_layout(n1, mma)
    n1 = gl.amd.cdna4.mfma(p_lo, v1, n1)
    v2 = gl.convert_layout(v2, dot_b).to(gl.bfloat16)
    n1 = gl.amd.cdna4.mfma(p_hi, v1, n1)
    n2 = gl.convert_layout(n2, mma)
    n2 = gl.amd.cdna4.mfma(p_lo, v2, n2)
    v3 = gl.convert_layout(v3, dot_b).to(gl.bfloat16)
    n2 = gl.amd.cdna4.mfma(p_hi, v2, n2)
    n3 = gl.convert_layout(n3, mma)
    n3 = gl.amd.cdna4.mfma(p_lo, v3, n3)
    v4 = gl.convert_layout(v4, dot_b).to(gl.bfloat16)
    n3 = gl.amd.cdna4.mfma(p_hi, v3, n3)
    n4 = gl.convert_layout(n4, mma)
    n4 = gl.amd.cdna4.mfma(p_lo, v4, n4)
    v5 = gl.convert_layout(v5, dot_b).to(gl.bfloat16)
    n4 = gl.amd.cdna4.mfma(p_hi, v4, n4)
    n5 = gl.convert_layout(n5, mma)
    n5 = gl.amd.cdna4.mfma(p_lo, v5, n5)
    v6 = gl.convert_layout(v6, dot_b).to(gl.bfloat16)
    n5 = gl.amd.cdna4.mfma(p_hi, v5, n5)
    n6 = gl.convert_layout(n6, mma)
    n6 = gl.amd.cdna4.mfma(p_lo, v6, n6)
    v7 = gl.convert_layout(v7, dot_b).to(gl.bfloat16)
    n6 = gl.amd.cdna4.mfma(p_hi, v6, n6)
    n7 = gl.convert_layout(n7, mma)
    n7 = gl.amd.cdna4.mfma(p_lo, v7, n7)
    n7 = gl.amd.cdna4.mfma(p_hi, v7, n7)
    result = _join_feature_eighths(n0, n1, n2, n3, n4, n5, n6, n7)
    return gl.convert_layout(result, mma)


@gluon.jit
def _local_peak(numerator):
    grouped = numerator.reshape((16, numerator.shape[1] // 64, 4, 16))
    return gl.max(gl.abs(grouped), 1)


@gluon.jit
def _two_tile_values(
    probability,
    key0,
    key1,
    Compact,
    record,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    mma: gl.constexpr,
):
    p_hi = probability.to(gl.bfloat16)
    p_lo = (probability - p_hi.to(gl.float32)).to(gl.bfloat16)
    lo0, lo1 = p_lo.reshape((16, 2, 64)).permute((0, 2, 1)).split()
    hi0, hi1 = p_hi.reshape((16, 2, 64)).permute((0, 2, 1)).split()
    hi0_shared = _stage_matrix(hi0, 8)
    lo0_shared = _stage_matrix(lo0, 8)
    hi1_shared = _stage_matrix(hi1, 8)
    lo1_shared = _stage_matrix(lo1, 8)
    lo0 = lo0_shared.load(dot_a)
    hi0 = hi0_shared.load(dot_a)
    lo1 = lo1_shared.load(dot_a)
    hi1 = hi1_shared.load(dot_a)
    value0 = gl.amd.cdna4.async_copy.load_shared_relaxed(key0, dot_b)
    value1 = gl.amd.cdna4.async_copy.load_shared_relaxed(key1, dot_b)
    value0_left, value0_right = value0.reshape((64, 2, 256)).permute((0, 2, 1)).split()
    value1_left, value1_right = value1.reshape((64, 2, 256)).permute((0, 2, 1)).split()
    value0_first, value0_middle_left = (
        value0_left.reshape((64, 2, 128)).permute((0, 2, 1)).split()
    )
    value0_middle_right, value0_last = (
        value0_right.reshape((64, 2, 128)).permute((0, 2, 1)).split()
    )
    value1_first, value1_middle_left = (
        value1_left.reshape((64, 2, 128)).permute((0, 2, 1)).split()
    )
    value1_middle_right, value1_last = (
        value1_right.reshape((64, 2, 128)).permute((0, 2, 1)).split()
    )
    value0_middle = (
        gl.join(value0_middle_left, value0_middle_right)
        .permute((0, 2, 1))
        .reshape((64, 256))
    )
    value1_middle = (
        gl.join(value1_middle_left, value1_middle_right)
        .permute((0, 2, 1))
        .reshape((64, 256))
    )
    n_first = gl.full((16, 128), 0.0, gl.float32, mma)
    n_middle = gl.full((16, 256), 0.0, gl.float32, mma)
    n_last = gl.full((16, 128), 0.0, gl.float32, mma)
    features128 = gl.arange(0, 128, gl.SliceLayout(0, mma))
    features256 = gl.arange(0, 256, gl.SliceLayout(0, mma))
    n_first = _accumulate_value_group(lo0, hi0, value0_first, n_first, dot_b, mma)
    n_first = _accumulate_value_group(lo1, hi1, value1_first, n_first, dot_b, mma)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + features128[None, :], n_first)
    local_peak = _local_peak(n_first)
    n_middle = _accumulate_value_group(lo0, hi0, value0_middle, n_middle, dot_b, mma)
    n_middle = _accumulate_value_group(lo1, hi1, value1_middle, n_middle, dot_b, mma)
    gl.store(
        Compact + record[:, None] * _PARTIAL_PITCH + 128 + features256[None, :],
        n_middle,
    )
    local_peak = gl.maximum(
        local_peak,
        gl.convert_layout(
            _local_peak(n_middle), local_peak.type.layout, assert_trivial=True
        ),
    )
    n_last = _accumulate_value_group(lo0, hi0, value0_last, n_last, dot_b, mma)
    n_last = _accumulate_value_group(lo1, hi1, value1_last, n_last, dot_b, mma)
    gl.store(
        Compact + record[:, None] * _PARTIAL_PITCH + 384 + features128[None, :], n_last
    )
    local_peak = gl.maximum(
        local_peak,
        gl.convert_layout(
            _local_peak(n_last), local_peak.type.layout, assert_trivial=True
        ),
    )
    n_middle0, n_middle1 = n_middle.reshape((16, 2, 128)).permute((0, 2, 1)).split()
    n_middle0 = gl.convert_layout(n_middle0, mma)
    n_middle1 = gl.convert_layout(n_middle1, mma)
    n0 = gl.join(n_first, n_middle0).permute((0, 2, 1)).reshape((16, 256))
    n1 = gl.join(n_middle1, n_last).permute((0, 2, 1)).reshape((16, 256))
    result = gl.join(n0, n1).permute((0, 2, 1)).reshape((16, 512))
    return (gl.convert_layout(result, mma), gl.max(local_peak, 2))


@gluon.jit
def _last_tile_values(
    p_lo,
    p_hi,
    value,
    numerator,
    Compact,
    record,
    dot_b: gl.constexpr,
    mma: gl.constexpr,
):
    v0, v1, v2, v3, v4, v5, v6, v7 = _split_feature_eighths(value)
    n0, n1, n2, n3, n4, n5, n6, n7 = _split_feature_eighths(numerator)
    features = gl.arange(0, 64, gl.SliceLayout(0, mma))
    v0 = gl.convert_layout(v0, dot_b).to(gl.bfloat16)
    n0 = gl.convert_layout(n0, mma)
    n0 = gl.amd.cdna4.mfma(p_lo, v0, n0)
    v1 = gl.convert_layout(v1, dot_b).to(gl.bfloat16)
    n0 = gl.amd.cdna4.mfma(p_hi, v0, n0)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 0 + features[None, :], n0)
    peak = _local_peak(n0)
    n1 = gl.convert_layout(n1, mma)
    n1 = gl.amd.cdna4.mfma(p_lo, v1, n1)
    v2 = gl.convert_layout(v2, dot_b).to(gl.bfloat16)
    n1 = gl.amd.cdna4.mfma(p_hi, v1, n1)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 64 + features[None, :], n1)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n1), peak.type.layout, assert_trivial=True)
    )
    n2 = gl.convert_layout(n2, mma)
    n2 = gl.amd.cdna4.mfma(p_lo, v2, n2)
    v3 = gl.convert_layout(v3, dot_b).to(gl.bfloat16)
    n2 = gl.amd.cdna4.mfma(p_hi, v2, n2)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 128 + features[None, :], n2)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n2), peak.type.layout, assert_trivial=True)
    )
    n3 = gl.convert_layout(n3, mma)
    n3 = gl.amd.cdna4.mfma(p_lo, v3, n3)
    v4 = gl.convert_layout(v4, dot_b).to(gl.bfloat16)
    n3 = gl.amd.cdna4.mfma(p_hi, v3, n3)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 192 + features[None, :], n3)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n3), peak.type.layout, assert_trivial=True)
    )
    n4 = gl.convert_layout(n4, mma)
    n4 = gl.amd.cdna4.mfma(p_lo, v4, n4)
    v5 = gl.convert_layout(v5, dot_b).to(gl.bfloat16)
    n4 = gl.amd.cdna4.mfma(p_hi, v4, n4)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 256 + features[None, :], n4)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n4), peak.type.layout, assert_trivial=True)
    )
    n5 = gl.convert_layout(n5, mma)
    n5 = gl.amd.cdna4.mfma(p_lo, v5, n5)
    v6 = gl.convert_layout(v6, dot_b).to(gl.bfloat16)
    n5 = gl.amd.cdna4.mfma(p_hi, v5, n5)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 320 + features[None, :], n5)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n5), peak.type.layout, assert_trivial=True)
    )
    n6 = gl.convert_layout(n6, mma)
    n6 = gl.amd.cdna4.mfma(p_lo, v6, n6)
    v7 = gl.convert_layout(v7, dot_b).to(gl.bfloat16)
    n6 = gl.amd.cdna4.mfma(p_hi, v6, n6)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 384 + features[None, :], n6)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n6), peak.type.layout, assert_trivial=True)
    )
    n7 = gl.convert_layout(n7, mma)
    n7 = gl.amd.cdna4.mfma(p_lo, v7, n7)
    n7 = gl.amd.cdna4.mfma(p_hi, v7, n7)
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + 448 + features[None, :], n7)
    peak = gl.maximum(
        peak, gl.convert_layout(_local_peak(n7), peak.type.layout, assert_trivial=True)
    )
    result = _join_feature_eighths(n0, n1, n2, n3, n4, n5, n6, n7)
    return (gl.convert_layout(result, mma), gl.max(peak, 2))


@gluon.jit
def _copy_m32_query(Q, row, Q0: gl.constexpr, Q1: gl.constexpr):
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    rotary_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    hh = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    dd = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    rh = gl.arange(0, 16, gl.SliceLayout(1, rotary_layout))
    rr = gl.arange(0, 64, gl.SliceLayout(0, rotary_layout))
    query_shared = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [16, 512],
        gl.PaddedSharedLayout.with_identity_for([[512, 16]], [16, 512], [1, 0]),
    )
    rotary_shared = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [16, 64],
        gl.PaddedSharedLayout.with_identity_for([[512, 16]], [16, 64], [1, 0]),
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        query_shared, Q + row * Q0, hh[:, None] * Q1 + dd[None, :]
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        rotary_shared, Q + row * Q0, rh[:, None] * Q1 + 512 + rr[None, :]
    )
    gl.amd.cdna4.async_copy.commit_group()
    return (query_shared, rotary_shared)


@gluon.jit
def _m32_any_slots(raw):
    return gl.max(raw, 0) >= 0


@gluon.jit
def _split_attention_m32(
    Q,
    KV,
    S,
    Numerator,
    Denominator,
    Maximum,
    WideNumerator,
    RecordFlag,
    scale,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
):
    Numerator = Numerator.to(gl.pointer_type(gl.float16))
    RecordFlag = RecordFlag.to(gl.pointer_type(gl.int32))
    row = gl.program_id(0)
    split = gl.program_id(1)
    load_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [32, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[64, 512],
    )
    rotary_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [32, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[64, 64],
    )
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    dd = gl.arange(0, 512, gl.SliceLayout(0, load_layout))
    rr = gl.arange(0, 64, gl.SliceLayout(0, rotary_layout))
    nn = gl.arange(0, 64, gl.SliceLayout(1, load_layout))
    rn = gl.arange(0, 64, gl.SliceLayout(1, rotary_layout))
    sn = gl.arange(0, 128, gl.SliceLayout(0, mma_layout))
    nh = gl.arange(0, 16, gl.SliceLayout(1, mma_layout))
    nv = gl.arange(0, 512, gl.SliceLayout(0, mma_layout))
    slots_layout: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    pos = split * 128 + gl.arange(0, 128, slots_layout)
    raw = _load_slots(S, row, pos, S0, SELECTED)
    offsets = gl.where(raw >= 0, raw.to(gl.int32) * (K0 // 16), -(1 << 27))
    slots_shared = gl.allocate_shared_memory(
        gl.int32, [128], gl.SwizzledSharedLayout(1, 1, 1, [0]), value=offsets
    )
    query_shared, rotary_query_shared = _copy_m32_query(Q, row, Q0, Q1)
    any_tokens = _m32_any_slots(raw)
    record = (row * SPLITS + split) * 16 + nh
    if any_tokens:
        offset0 = slots_shared.gather(nn, 0)
        rotary_offset0 = slots_shared.gather(rn, 0)
        key0, rotary0 = _allocate_m64_kv(KV.dtype.element_ty)
        _copy_m32_kv(KV, key0, rotary0, offset0, rotary_offset0, dd, rr)
        gl.amd.cdna4.async_copy.wait_group(1)
        qq = query_shared.load(dot_a)
        qr = rotary_query_shared.load(dot_a)
        offset1 = slots_shared.gather(nn + 64, 0)
        rotary_offset1 = slots_shared.gather(rn + 64, 0)
        key1, rotary1 = _allocate_m64_kv(KV.dtype.element_ty)
        _copy_m32_kv(KV, key1, rotary1, offset1, rotary_offset1, dd, rr)
        active = slots_shared.gather(sn, 0) >= 0
        gl.amd.cdna4.async_copy.wait_group(1)
        score0 = _tile_score(qq, qr, key0, rotary0, dot_b, mma_layout)
        gl.amd.cdna4.async_copy.wait_group(0)
        score1 = _tile_score(qq, qr, key1, rotary1, dot_b, mma_layout)
        score = gl.join(score0, score1).permute((0, 2, 1)).reshape((16, 128))
        score = gl.convert_layout(score, mma_layout)
        score = gl.where(active[None, :], score * scale, -float("inf"))
        maximum = gl.max(score, 1)
        probability = gl.where(
            active[None, :], _natural_exp(score - maximum[:, None]), 0.0
        )
        den_partial = gl.sum(gl.sum(probability.reshape((16, 2, 4, 16)), 1), 2)
        den_shared = gl.allocate_shared_memory(
            gl.float32,
            [16, 4],
            gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
            value=den_partial,
        )
        numerator, peak = _two_tile_values(
            probability, key0, key1, Numerator, record, dot_a, dot_b, mma_layout
        )
        den_layout: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [4, 1], [0, 1])
        den_parts = gl.amd.cdna4.async_copy.load_shared_relaxed(den_shared, den_layout)
        denominator = gl.convert_layout(
            gl.sum(den_parts, 1), gl.SliceLayout(1, mma_layout)
        )
        _store_partial_tags_and_wide(
            WideNumerator, RecordFlag, record, nv, numerator, denominator, peak
        )
        gl.store(Denominator + record, denominator)
        gl.store(Maximum + record, maximum)
    else:
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.store(RecordFlag + record, -1)
        gl.store(Denominator + record, 0.0)
        gl.store(Maximum + record, -float("inf"))


@gluon.jit
def _merge_m64_group(Numerator, WideNumerator, flag, alpha, record, vv):
    feature_flag = flag[:, :, None] >> vv[None, None, :] % 64 // 16 * 8 & 255
    feature_flag = gl.max_constancy(feature_flag, [1, 1, 16])
    group_base = record[:, :, None] * _PARTIAL_PITCH
    compact_base = gl.where(feature_flag == 0, group_base, -(1 << 30))
    compact_offset = gl.max_contiguous(
        gl.multiple_of(compact_base, [1, 1, 8]) + vv[None, None, :], [1, 1, 8]
    )
    numerator = gl.amd.cdna4.buffer_load(Numerator, compact_offset).to(gl.float32)
    if gl.max(gl.max(flag, 1), 0) > 0:
        wide_base = gl.where(feature_flag == 1, group_base, -(1 << 29))
        wide_offset = gl.max_contiguous(
            gl.multiple_of(wide_base, [1, 1, 8]) + vv[None, None, :], [1, 1, 8]
        )
        wide = gl.amd.cdna4.buffer_load(WideNumerator, wide_offset)
        numerator = gl.where(feature_flag == 1, wide, numerator)
    return gl.sum(numerator * alpha[:, :, None], 1)


@gluon.jit
def _merge_attention(
    Numerator,
    Denominator,
    Maximum,
    WideNumerator,
    RecordFlag,
    O,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_S: gl.constexpr,
    SMALL_BATCH: gl.constexpr,
    HEADS: gl.constexpr,
):
    Numerator = Numerator.to(gl.pointer_type(gl.float16))
    RecordFlag = RecordFlag.to(gl.pointer_type(gl.int32))
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [HEADS, 2, 32 // HEADS], [1, 1, gl.num_warps()], [2, 1, 0]
    )
    stats_layout: gl.constexpr = gl.SliceLayout(2, layout)
    hh = gl.program_id(1) * HEADS + gl.arange(0, HEADS, gl.SliceLayout(1, stats_layout))
    ss = gl.arange(0, BLOCK_S, gl.SliceLayout(0, stats_layout))
    vv = gl.arange(0, 512, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    record = (row * SPLITS + ss[None, :]) * H + hh[:, None]
    STAT_PITCH: gl.constexpr = 1 if SMALL_BATCH else 4
    if SMALL_BATCH:
        flag = gl.load(RecordFlag + record, ss[None, :] < SPLITS, -1)
        maximum = gl.load(Maximum + record, ss[None, :] < SPLITS, -float("inf"))
        denominator = gl.load(Denominator + record, ss[None, :] < SPLITS, 0.0)
    else:
        head_origin = gl.program_id(1) * HEADS
        stats_origin = (row * SPLITS * H + head_origin) * STAT_PITCH
        local_head = gl.arange(0, HEADS, gl.SliceLayout(1, stats_layout))
        stats_offset = (ss[None, :] * H + local_head[:, None]) * STAT_PITCH
        flag = gl.amd.cdna4.buffer_load(
            RecordFlag + stats_origin, stats_offset, ss[None, :] < SPLITS, -1
        )
        maximum = gl.amd.cdna4.buffer_load(
            Maximum + stats_origin, stats_offset, ss[None, :] < SPLITS, -float("inf")
        )
        denominator = gl.amd.cdna4.buffer_load(
            Denominator + stats_origin, stats_offset, ss[None, :] < SPLITS, 0.0
        )
    head_flag = gl.max(flag, 1)
    max_flag = gl.max(head_flag, 0)
    if SMALL_BATCH:
        row_maximum = gl.max(maximum, 1)
        global_max = gl.where(row_maximum != -float("inf"), row_maximum, 0.0)
    else:
        global_max = gl.where(head_flag >= 0, gl.max(maximum, 1), 0.0)
    alpha = _natural_exp(maximum - global_max[:, None])
    denominator = gl.sum(denominator * alpha, 1)
    if not SMALL_BATCH:
        reciprocal = 1.0 / gl.where(denominator > 0, denominator, 1.0)
        alpha *= reciprocal[:, None]
    if not SMALL_BATCH and BLOCK_S == 16:
        flag0, flag1 = flag.reshape((HEADS, 2, 8)).permute((0, 2, 1)).split()
        alpha0, alpha1 = alpha.reshape((HEADS, 2, 8)).permute((0, 2, 1)).split()
        record0, record1 = record.reshape((HEADS, 2, 8)).permute((0, 2, 1)).split()
        flag0 = gl.convert_layout(flag0, stats_layout, assert_trivial=True)
        flag1 = gl.convert_layout(flag1, stats_layout, assert_trivial=True)
        alpha0 = gl.convert_layout(alpha0, stats_layout, assert_trivial=True)
        alpha1 = gl.convert_layout(alpha1, stats_layout, assert_trivial=True)
        record0 = gl.convert_layout(record0, stats_layout, assert_trivial=True)
        record1 = gl.convert_layout(record1, stats_layout, assert_trivial=True)
        flag1, _ = flag1.reshape((HEADS, 2, 4)).permute((0, 2, 1)).split()
        alpha1, _ = alpha1.reshape((HEADS, 2, 4)).permute((0, 2, 1)).split()
        record1, _ = record1.reshape((HEADS, 2, 4)).permute((0, 2, 1)).split()
        flag1 = gl.convert_layout(flag1, stats_layout, assert_trivial=True)
        alpha1 = gl.convert_layout(alpha1, stats_layout, assert_trivial=True)
        record1 = gl.convert_layout(record1, stats_layout, assert_trivial=True)
        numerator = _merge_m64_group(
            Numerator, WideNumerator, flag0, alpha0, record0, vv
        )
        head_tail = gl.max(flag1, 1)
        first_tail = gl.gather(
            head_tail, gl.full((1,), 0, gl.int32, head_tail.type.layout), 0
        )
        if gl.sum(first_tail, 0) >= 0:
            numerator += _merge_m64_group(
                Numerator, WideNumerator, flag1, alpha1, record1, vv
            )
    else:
        feature_flag = flag[:, :, None] >> vv[None, None, :] % 64 // 16 * 8 & 255
        feature_flag = gl.max_constancy(feature_flag, [1, 1, 16])
        group_base = record[:, :, None] * _PARTIAL_PITCH
        compact_base = gl.where(feature_flag == 0, group_base, -(1 << 30))
        compact_offset = gl.max_contiguous(
            gl.multiple_of(compact_base, [1, 1, 8]) + vv[None, None, :], [1, 1, 8]
        )
        numerator = gl.amd.cdna4.buffer_load(Numerator, compact_offset).to(gl.float32)
        if max_flag > 0:
            wide_base = gl.where(feature_flag == 1, group_base, -(1 << 29))
            wide_offset = gl.max_contiguous(
                gl.multiple_of(wide_base, [1, 1, 8]) + vv[None, None, :], [1, 1, 8]
            )
            wide_numerator = gl.amd.cdna4.buffer_load(WideNumerator, wide_offset)
            numerator = gl.where(feature_flag == 1, wide_numerator, numerator)
        numerator = gl.sum(numerator * alpha[:, :, None], 1)
    if SMALL_BATCH:
        inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
        inverse_denominator = gl.convert_layout(
            inverse_denominator, gl.SliceLayout(1, numerator.type.layout)
        )
        result = numerator * inverse_denominator[:, None]
    else:
        result = numerator
    oh = gl.convert_layout(hh, gl.SliceLayout(1, result.type.layout))
    ov = gl.convert_layout(vv, gl.SliceLayout(0, result.type.layout))
    gl.store(O + (row * H + oh[:, None]) * 512 + ov[None, :], result.to(gl.bfloat16))


@gluon.jit
def _merge_attention_m64(
    Workspace,
    O,
    ROWS: gl.constexpr,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_S: gl.constexpr,
    HEADS: gl.constexpr,
):
    PAYLOAD: gl.constexpr = ROWS * SPLITS * H * _PARTIAL_PITCH
    compact = Workspace.to(gl.pointer_type(gl.float16))
    wide = Workspace + PAYLOAD // 2
    flags = (Workspace + 3 * PAYLOAD // 2).to(gl.pointer_type(gl.int32))
    den = Workspace + 3 * PAYLOAD // 2 + 2
    maximum = Workspace + 3 * PAYLOAD // 2 + 3
    _merge_attention(
        compact, den, maximum, wide, flags, O, H, SPLITS, BLOCK_S, False, HEADS
    )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, h, _ = query.shape
    assert h == 16
    selected = selected_slots.shape[1]
    small_batch = m < 48
    chunk = 128 if small_batch else 192
    splits = triton.cdiv(selected, chunk)
    cache_stride = kv_cache.stride(0)
    cache_span = (
        (kv_cache.shape[0] - 1) * cache_stride + kv_cache.shape[1]
    ) * kv_cache.element_size()
    cache_aligned = (
        cache_stride > 0
        and cache_stride * kv_cache.element_size() % 16 == 0
        and (kv_cache.storage_offset() * kv_cache.element_size() % 16 == 0)
    )
    assert cache_aligned and 0 < cache_stride <= 2**31 - 1
    assert 0 < cache_span <= 2**31 - 1
    assert m * splits * h * _PARTIAL_PITCH.value * 4 <= 2**31 - 1
    records = m * splits * h
    payload_elements = records * _PARTIAL_PITCH.value
    wide_start = payload_elements // 2
    stats_start = wide_start + payload_elements
    if small_batch:
        stats_pitch = triton.cdiv(records, 64) * 64
        den_start = stats_start + stats_pitch
        max_start = den_start + stats_pitch
        workspace = query.new_empty((max_start + stats_pitch,), dtype=torch.float32)
        record_flag = workspace[stats_start:]
        denominator = workspace[den_start:]
        maximum = workspace[max_start:]
        numerator = workspace
        wide_numerator = workspace[wide_start:]
        output = query.new_empty((m, h, 512), dtype=torch.bfloat16)
        _split_attention_m32[m, splits](
            query,
            kv_cache,
            selected_slots,
            numerator,
            denominator,
            maximum,
            wide_numerator,
            record_flag,
            softmax_scale,
            query.stride(0),
            query.stride(1),
            kv_cache.stride(0),
            selected_slots.stride(0),
            selected,
            splits,
            num_warps=4,
            waves_per_eu=2,
        )
        _merge_attention[m, h // 2](
            numerator,
            denominator,
            maximum,
            wide_numerator,
            record_flag,
            output,
            h,
            splits,
            triton.next_power_of_2(splits),
            SMALL_BATCH=True,
            HEADS=2,
            num_warps=4,
        )
    else:
        workspace = query.new_empty((stats_start + records * 4,), dtype=torch.float32)
        output = query.new_empty((m, h, 512), dtype=torch.bfloat16)
        _split_attention_m64[m, splits](
            query,
            kv_cache,
            selected_slots,
            workspace,
            softmax_scale,
            m,
            query.stride(0),
            query.stride(1),
            kv_cache.stride(0),
            selected_slots.stride(0),
            selected,
            splits,
            num_warps=4,
            waves_per_eu=2,
        )
        _merge_attention_m64[m, h // 4](
            workspace,
            output,
            m,
            h,
            splits,
            triton.next_power_of_2(splits),
            HEADS=4,
            num_warps=4,
            waves_per_eu=2,
        )
    return output
