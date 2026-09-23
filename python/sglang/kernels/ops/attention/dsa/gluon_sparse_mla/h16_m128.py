# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m128.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_PARTIAL_PITCH = gl.constexpr(520)
_STATS_PITCH = gl.constexpr(260)


@gluon.constexpr_function
def _m64_kv_layouts(width):
    vector_bits = [[0, 1], [0, 2], [0, 4], [0, 8]]
    warp_bits = [[1, 0], [2, 0]]
    if width == 256 or width == 512:
        lane_bits = [[0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]]
        register_bits = [[4, 0], [8, 0]]
        if width == 512:
            register_bits += [[0, 256]]
    else:
        assert width == 64
        lane_bits = [[0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [32, 0]]
        register_bits = []
    shape = [64, width]
    distributed = gl.DistributedLinearLayout(
        reg_bases=vector_bits + register_bits,
        lane_bases=lane_bits,
        warp_bases=warp_bits,
        block_bases=[],
        shape=shape,
    )
    shared = gl.PaddedSharedLayout(
        [[1024, 8]], vector_bits + lane_bits + warp_bits + register_bits, [], shape
    )
    return (distributed, shared)


@gluon.jit
def _stage_padded(value, padding: gl.constexpr):
    return gl.allocate_shared_memory(
        value.dtype,
        value.shape,
        gl.PaddedSharedLayout.with_identity_for(
            [[value.shape[1], padding]], value.shape, [1, 0]
        ),
        value=value,
    )


@gluon.jit
def _natural_exp(x):
    return gl.exp2(x * 1.4426950408889634)


@gluon.jit
def _bitwise_union(a, b):
    return a | b


@gluon.jit
def _sum_counts(a, b):
    return a + b


@gluon.jit
def _schedule_summary(a_extent, a_count, b_extent, b_count):
    return (gl.maximum(a_extent, b_extent), a_count + b_count)


@gluon.jit
def _first_live_tile(bits):
    return gl.inline_asm_elementwise(
        "v_ffbl_b32 $0, $1",
        constraints="=v,v",
        args=[bits],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _encode_slot(slot, K0: gl.constexpr):
    return gl.where(slot >= 0, slot.to(gl.int32) * (K0 // 16), -134217728)


@gluon.jit
def _prefetch_fragment(
    KV, shared, slot, channels, start: gl.constexpr, CACHE_POLICY: gl.constexpr
):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        shared,
        KV,
        slot[:, None] * 16 + start + channels[None, :],
        cache_modifier=CACHE_POLICY,
    )


@gluon.jit
def _packed_pv(probability, value, accumulator, dot_b: gl.constexpr):
    value = gl.join(value, value).permute((2, 0, 1)).reshape((128, value.shape[1]))
    value = gl.convert_layout(value, dot_b)
    return gl.amd.cdna4.mfma(probability, value, accumulator)


@gluon.jit
def _store_numerator(
    Compact, Wide, Statistics, record, features, numerator, denominator
):
    striped = (
        numerator.reshape((16, 8, 4, 16)).permute((0, 2, 1, 3)).reshape((16, 4, 128))
    )
    peak = gl.max(gl.abs(striped), 2)
    flag_layout: gl.constexpr = peak.type.layout
    local_den = gl.convert_layout(denominator, gl.SliceLayout(1, flag_layout))
    wide = peak > local_den[:, None] * 0.5
    local_record = gl.convert_layout(record, gl.SliceLayout(1, flag_layout))
    group = gl.arange(0, 4, gl.SliceLayout(0, flag_layout))
    tag_bytes = Statistics.to(gl.pointer_type(gl.int8))
    gl.store(
        tag_bytes + local_record[:, None] * (_PARTIAL_PITCH * 2) + 8 + group[None, :],
        wide.to(gl.int8),
    )
    wide_features = gl.broadcast(wide[:, :, None], striped)[0]
    wide_features = (
        wide_features.reshape((16, 4, 8, 16)).permute((0, 2, 1, 3)).reshape((16, 512))
    )
    wide_features = gl.convert_layout(wide_features, numerator.type.layout)
    wide_features = gl.max_constancy(wide_features, [1, 16])
    gl.store(Compact + record[:, None] * _PARTIAL_PITCH + features[None, :], numerator)
    gl.store(
        Wide + record[:, None] * _PARTIAL_PITCH + features[None, :],
        numerator,
        wide_features,
    )


@gluon.jit
def _load_mixed_numerator(numerator, Wide, record, features, flag):
    feature_flag = flag[:, None] >> features[None, :] % 64 // 16 * 8 & 255
    feature_flag = gl.max_constancy(feature_flag, [1, 16])
    wide_numerator = gl.amd.cdna4.buffer_load(
        Wide, record[:, None] * _PARTIAL_PITCH + features[None, :], feature_flag == 1, 0
    )
    return gl.where(feature_flag == 1, wide_numerator, numerator)


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
def _stage_query_halves(qq, qr, dot_a: gl.constexpr):
    q0, q1 = qq.reshape((16, 2, 256)).permute((0, 2, 1)).split()
    q0_shared = _stage_padded(q0, 16)
    q0 = q0_shared.load(dot_a)
    q1_shared = _stage_padded(q1, 16)
    q1 = q1_shared.load(dot_a)
    qr_shared = _stage_padded(qr, 16)
    qr = qr_shared.load(dot_a)
    return (q0, q1, qr)


@gluon.jit
def _merge_attention(
    Numerator,
    WideNumerator,
    Statistics,
    O,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_S: gl.constexpr,
):
    row = gl.program_id(0)
    head = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [1, 1], [1, 0])
    ss = gl.arange(0, BLOCK_S, gl.SliceLayout(1, layout))
    vv = gl.arange(0, 512, gl.SliceLayout(0, layout))
    record = (row * SPLITS + ss) * H + head
    numerator = gl.amd.cdna4.buffer_load(
        Numerator,
        record[:, None] * _PARTIAL_PITCH + vv[None, :],
        ss[:, None] < SPLITS,
        0,
    ).to(gl.float32)
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, 1], [1, 0])
    stats_ss = gl.arange(0, BLOCK_S, gl.SliceLayout(1, stats_layout))
    stats_field = gl.arange(0, 4, gl.SliceLayout(0, stats_layout))
    stats_record = (row * SPLITS + stats_ss) * H + head
    stats = gl.load(
        Statistics + stats_record[:, None] * _STATS_PITCH + stats_field[None, :],
        stats_ss[:, None] < SPLITS,
        0,
    )
    first, second = stats.reshape((BLOCK_S, 2, 2)).permute((0, 2, 1)).split()
    denominator, maximum = first.split()
    flag, _ = second.split()
    denominator = gl.convert_layout(denominator, gl.SliceLayout(1, layout))
    maximum = gl.convert_layout(maximum, gl.SliceLayout(1, layout))
    flag = gl.convert_layout(flag.to(gl.int32, bitcast=True), gl.SliceLayout(1, layout))
    flag = gl.where(ss < SPLITS, flag, -1)
    maximum = gl.where(ss < SPLITS, maximum, -float("inf"))
    max_flag = gl.max(flag, 0)
    global_max = gl.where(max_flag >= 0, gl.max(maximum, 0), 0.0)
    alpha = _natural_exp(maximum - global_max)
    denominator = gl.sum(denominator * alpha, 0)
    if max_flag > 0:
        numerator = _load_mixed_numerator(numerator, WideNumerator, record, vv, flag)
    numerator = gl.sum(numerator * alpha[:, None], 0)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    gl.store(
        O + (row * H + head) * 512 + vv,
        (numerator * inverse_denominator).to(gl.bfloat16),
    )


@gluon.jit
def _attention_tile(
    KV,
    schedule,
    key_shared,
    rotary_shared,
    q0,
    q1,
    qr,
    maximum,
    denominator_terms,
    numerator0,
    numerator1,
    tile,
    next_tile,
    has_next,
    nn,
    rn,
    sn,
    dd,
    rr,
    scale,
    mma: gl.constexpr,
    dot_a: gl.constexpr,
    dot_b: gl.constexpr,
    CHUNK: gl.constexpr,
    SLOT_BLOCK: gl.constexpr,
    CACHE_POLICY: gl.constexpr,
    DEFER_P_LOAD: gl.constexpr,
):
    active = schedule.gather(tile * 64 + sn, axis=0) >= 0
    next_slot = schedule.gather(next_tile * 64 + nn & SLOT_BLOCK - 1, axis=0)
    next_rotary_slot = schedule.gather(next_tile * 64 + rn & SLOT_BLOCK - 1, axis=0)
    gl.amd.cdna4.async_copy.wait_group(0)
    k0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_shared.slice(0, 256, 1).permute((1, 0)), dot_b
    ).to(gl.bfloat16)
    zero = gl.full((16, 64), 0, gl.float32, mma)
    score0 = gl.amd.cdna4.mfma(q0, k0, zero)
    k1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_shared.slice(256, 256, 1).permute((1, 0)), dot_b
    ).to(gl.bfloat16)
    score1 = gl.amd.cdna4.mfma(q1, k1, zero)
    rotary_t = gl.amd.cdna4.async_copy.load_shared_relaxed(
        rotary_shared.permute((1, 0)), dot_b
    ).to(gl.bfloat16)
    score1 = gl.amd.cdna4.mfma(qr, rotary_t, score1)
    scores = (score0 + score1) * scale
    scores = gl.where(active[None, :], scores, -float("inf"))
    next_max = gl.maximum(maximum, gl.max(scores, 1))
    probability = gl.where(
        active[None, :], _natural_exp(scores - next_max[:, None]), 0.0
    )
    probability_terms = gl.sum(probability.reshape((16, 4, 16)), 2)
    if tile == 0:
        denominator_terms = probability_terms
    else:
        alpha = gl.where(
            next_max == -float("inf"), 1.0, _natural_exp(maximum - next_max)
        )
        den_alpha = gl.convert_layout(
            alpha, gl.SliceLayout(1, denominator_terms.type.layout)
        )
        denominator_terms = denominator_terms * den_alpha[:, None] + probability_terms
        numerator0 *= alpha[:, None]
        numerator1 *= alpha[:, None]
    p_hi = probability.to(gl.bfloat16)
    p_lo = (probability - p_hi.to(gl.float32)).to(gl.bfloat16)
    packed = gl.join(p_lo, p_hi).permute((0, 2, 1)).reshape((16, 128))
    p_shared = _stage_padded(packed, 8)
    if not DEFER_P_LOAD:
        packed = p_shared.load(dot_a)
    value = gl.amd.cdna4.async_copy.load_shared_relaxed(key_shared, dot_b).to(
        gl.bfloat16
    )
    value0, value1 = value.reshape((64, 2, 256)).permute((0, 2, 1)).split()
    value0 = gl.convert_layout(value0, dot_b)
    value1 = gl.convert_layout(value1, dot_b)
    gl.barrier()
    if has_next:
        _prefetch_fragment(KV, key_shared, next_slot, dd, 0, CACHE_POLICY)
        _prefetch_fragment(KV, rotary_shared, next_rotary_slot, rr, 512, CACHE_POLICY)
        gl.amd.cdna4.async_copy.commit_group()
    if DEFER_P_LOAD:
        packed = p_shared.load(dot_a)
    numerator0 = _packed_pv(packed, value0, numerator0, dot_b)
    numerator1 = _packed_pv(packed, value1, numerator1, dot_b)
    return (next_max, denominator_terms, numerator0, numerator1)


@gluon.jit
def _chunk_attention(
    Q,
    KV,
    S,
    Compact,
    Wide,
    Statistics,
    scale,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    H: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    CHUNK: gl.constexpr,
    SLOT_BLOCK: gl.constexpr,
    CACHE_POLICY: gl.constexpr,
    DEFER_P_LOAD: gl.constexpr,
    M: gl.constexpr,
    ROW_GROUP: gl.constexpr,
    STRIPE: gl.constexpr,
):
    gl.static_assert(CHUNK % 64 == 0 and CHUNK <= SLOT_BLOCK)
    if ROW_GROUP == 1:
        row = gl.program_id(0)
        split = gl.program_id(1)
    elif M % ROW_GROUP == 0:
        row = gl.program_id(0) + ROW_GROUP * gl.program_id(2)
        split = gl.program_id(1)
    else:
        pid = gl.program_id(0)
        group_start = pid // (ROW_GROUP * SPLITS) * ROW_GROUP
        group_size = gl.minimum(M - group_start, ROW_GROUP)
        within_group = pid % (ROW_GROUP * SPLITS)
        row = group_start + within_group % group_size
        split = within_group // group_size
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    KEY_WIDTH: gl.constexpr = 512
    key_layouts: gl.constexpr = _m64_kv_layouts(KEY_WIDTH)
    rotary_layouts: gl.constexpr = _m64_kv_layouts(64)
    load_layout: gl.constexpr = key_layouts[0]
    rotary_layout: gl.constexpr = rotary_layouts[0]
    nn = gl.arange(0, 64, gl.SliceLayout(1, load_layout))
    rn = gl.arange(0, 64, gl.SliceLayout(1, rotary_layout))
    sn = gl.arange(0, 64, gl.SliceLayout(0, mma))
    dd = gl.arange(0, KEY_WIDTH, gl.SliceLayout(0, load_layout))
    rr = gl.arange(0, 64, gl.SliceLayout(0, rotary_layout))
    hh = gl.arange(0, 16, gl.SliceLayout(1, mma))
    vv = gl.arange(0, 512, gl.SliceLayout(0, mma))
    query, rotary_query = _load_query_tiles(Q, row, Q0, Q1)
    pos = gl.arange(
        0, SLOT_BLOCK, gl.BlockedLayout([1 if M < 192 else 4], [64], [4], [0])
    )
    selected_pos = (pos // STRIPE * SPLITS + split) * STRIPE + pos % STRIPE
    if SELECTED == CHUNK * SPLITS and CHUNK == SLOT_BLOCK:
        slots = gl.load(S + row * S0 + selected_pos)
    else:
        slots = gl.load(
            S + row * S0 + selected_pos, (pos < CHUNK) & (selected_pos < SELECTED), -1
        )
    valid = slots >= 0
    if M >= 192:
        live_tiles = gl.reduce(gl.where(valid, 1 << pos // 64, 0), 0, _bitwise_union)
        any_tokens = live_tiles != 0
    else:
        last_live, live_count = gl.reduce(
            (gl.where(valid, pos + 1, 0), valid.to(gl.int32)), 0, _schedule_summary
        )
        tiles = gl.cdiv(last_live, 64)
        any_tokens = tiles > 0
    published = _encode_slot(slots, K0)
    schedule = gl.allocate_shared_memory(
        published.dtype, [SLOT_BLOCK], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    if M < 192:
        packed_tiles = gl.cdiv(live_count, 64)
        if packed_tiles < tiles:
            prefix = gl.associative_scan(valid.to(gl.int32), 0, _sum_counts)
            destination = gl.where(valid, prefix - 1, live_count + pos - prefix)
            schedule.scatter(published, destination, axis=0)
            tiles = packed_tiles
        else:
            schedule.store(published)
    else:
        schedule.store(published)
    gl.barrier()
    record = (row * SPLITS + split) * H + hh
    if any_tokens:
        key_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty, [64, KEY_WIDTH], key_layouts[1]
        )
        rotary_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty, [64, 64], rotary_layouts[1]
        )
        if M >= 192:
            tile = _first_live_tile(live_tiles)
        else:
            tile = 0
        initial_slot = schedule.gather(tile * 64 + nn, axis=0)
        initial_rotary_slot = schedule.gather(tile * 64 + rn, axis=0)
        _prefetch_fragment(KV, key_shared, initial_slot, dd, 0, CACHE_POLICY)
        _prefetch_fragment(
            KV, rotary_shared, initial_rotary_slot, rr, 512, CACHE_POLICY
        )
        gl.amd.cdna4.async_copy.commit_group()
        q0, q1, qr = _stage_query_halves(query, rotary_query, dot_a)
        maximum = gl.full((16,), -float("inf"), gl.float32, gl.SliceLayout(1, mma))
        denominator_terms = gl.full((16, 64), 0, gl.float32, mma)
        denominator_terms = gl.sum(denominator_terms.reshape((16, 4, 16)), 2)
        numerator0 = gl.full((16, 256), 0, gl.float32, mma)
        numerator1 = gl.full((16, 256), 0, gl.float32, mma)
        if M >= 192:
            while live_tiles != 0:
                remaining = live_tiles & live_tiles - 1
                next_tile = _first_live_tile(remaining)
                maximum, denominator_terms, numerator0, numerator1 = _attention_tile(
                    KV,
                    schedule,
                    key_shared,
                    rotary_shared,
                    q0,
                    q1,
                    qr,
                    maximum,
                    denominator_terms,
                    numerator0,
                    numerator1,
                    tile,
                    next_tile,
                    remaining != 0,
                    nn,
                    rn,
                    sn,
                    dd,
                    rr,
                    scale,
                    mma,
                    dot_a,
                    dot_b,
                    CHUNK,
                    SLOT_BLOCK,
                    CACHE_POLICY,
                    DEFER_P_LOAD,
                )
                tile = next_tile
                live_tiles = remaining
        else:
            for tile in range(tiles):
                maximum, denominator_terms, numerator0, numerator1 = _attention_tile(
                    KV,
                    schedule,
                    key_shared,
                    rotary_shared,
                    q0,
                    q1,
                    qr,
                    maximum,
                    denominator_terms,
                    numerator0,
                    numerator1,
                    tile,
                    tile + 1,
                    tile + 1 < tiles,
                    nn,
                    rn,
                    sn,
                    dd,
                    rr,
                    scale,
                    mma,
                    dot_a,
                    dot_b,
                    CHUNK,
                    SLOT_BLOCK,
                    CACHE_POLICY,
                    DEFER_P_LOAD,
                )
        numerator = (
            gl.join(numerator0, numerator1).permute((0, 2, 1)).reshape((16, 512))
        )
        numerator = gl.convert_layout(numerator, mma)
        denominator = gl.convert_layout(
            gl.sum(denominator_terms, 1), gl.SliceLayout(1, mma)
        )
        _store_numerator(Compact, Wide, Statistics, record, vv, numerator, denominator)
        gl.store(Statistics + record * _STATS_PITCH, denominator)
        gl.store(Statistics + record * _STATS_PITCH + 1, maximum)
        gl.store(Statistics + record * _STATS_PITCH + 3, 0.0)
    else:
        gl.store(Compact + record[:, None] * _PARTIAL_PITCH + vv[None, :], 0.0)
        gl.store(
            Statistics.to(gl.pointer_type(gl.int32)) + record * _STATS_PITCH + 2, -1
        )
        gl.store(Statistics + record * _STATS_PITCH, 0.0)
        gl.store(Statistics + record * _STATS_PITCH + 1, -float("inf"))
        gl.store(Statistics + record * _STATS_PITCH + 3, 0.0)


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, h, _ = query.shape
    chunk = 512 if m < 192 else 1024
    assert h == 16
    selected = selected_slots.shape[1]
    splits = triton.cdiv(selected, chunk)
    cache_stride = kv_cache.stride(0)
    assert cache_stride > 0 and cache_stride % 16 == 0
    assert kv_cache.storage_offset() % 16 == 0
    assert (kv_cache.shape[0] - 1) * cache_stride + 576 < 2**31 - 1
    records = m * splits * h
    record_shape = (m, splits, h)
    pitch = _PARTIAL_PITCH.value
    wide_words = records * pitch
    compact_end = wide_words + wide_words // 2
    storage = torch.empty((compact_end,), device=query.device, dtype=torch.float32)
    wide = storage[:wide_words].view(*record_shape, pitch)
    compact = (
        storage[wide_words:compact_end].view(torch.float16).view(*record_shape, pitch)
    )
    statistics = compact.view(torch.float32)[..., 256:]
    output = torch.empty((m, h, 512), device=query.device, dtype=torch.bfloat16)
    row_group = 4 if m < 192 else 1
    stripe = 64 if m < 192 else 256
    if row_group == 1:
        grid = (m, splits)
    elif m % row_group == 0:
        grid = (row_group, splits, m // row_group)
    else:
        grid = (m * splits,)
    _chunk_attention[grid](
        query,
        kv_cache,
        selected_slots,
        compact,
        wide,
        statistics,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        cache_stride,
        selected_slots.stride(0),
        h,
        selected,
        splits,
        chunk,
        triton.next_power_of_2(chunk),
        ".ca",
        True,
        m,
        row_group,
        stripe,
        num_warps=4,
        waves_per_eu=2,
    )
    _merge_attention[m, h](
        compact,
        wide,
        statistics,
        output,
        h,
        splits,
        triton.next_power_of_2(splits),
        num_warps=1,
    )
    return output
