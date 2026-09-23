# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m2.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _exp(x):
    return gl.exp(x)


@gluon.jit
def _cache_load(cache, offsets, mask, BUFFER: gl.constexpr):
    if BUFFER:
        return gl.amd.cdna4.buffer_load(cache, offsets.to(gl.int32), mask, 0.0)
    return gl.load(cache + offsets, mask, 0.0)


@gluon.jit
def _value_fragment(
    shared,
    group,
    WIDTH: gl.constexpr,
    LAYOUT: gl.constexpr,
    COMMON_LOAD: gl.constexpr = False,
):
    if COMMON_LOAD:
        gl.static_assert(WIDTH == 128)
        if group == 0:
            fragment = shared.slice(0, 128, 1)
        elif group == 1:
            fragment = shared.slice(128, 128, 1)
        elif group == 2:
            fragment = shared.slice(256, 128, 1)
        else:
            fragment = shared.slice(384, 128, 1)
        return fragment.load(LAYOUT).to(gl.bfloat16)
    if WIDTH == 512:
        value = shared.load(LAYOUT)
    elif WIDTH == 256:
        if group == 0:
            value = shared.slice(0, 256, 1).load(LAYOUT)
        else:
            value = shared.slice(256, 256, 1).load(LAYOUT)
    elif group == 0:
        value = shared.slice(0, 128, 1).load(LAYOUT)
    elif group == 1:
        value = shared.slice(128, 128, 1).load(LAYOUT)
    elif group == 2:
        value = shared.slice(256, 128, 1).load(LAYOUT)
    else:
        value = shared.slice(384, 128, 1).load(LAYOUT)
    return value.to(gl.bfloat16)


@gluon.jit
def _store_fragment(
    P,
    numerator,
    row,
    part,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    FRAGMENT: gl.constexpr,
    WIDTH: gl.constexpr,
    BUFFER: gl.constexpr,
):
    layout: gl.constexpr = numerator.type.layout
    heads = gl.arange(0, 8, gl.SliceLayout(1, layout))
    dims = FRAGMENT * WIDTH + gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    offset = ((row * H + heads[:, None]) * SPLITS + part) * 528 + dims[None, :]
    if BUFFER:
        gl.amd.cdna4.buffer_store(numerator, P, offset.to(gl.int32), heads[:, None] < H)
    else:
        gl.store(P + offset, numerator, heads[:, None] < H)


@gluon.jit
def _copy_offsets(
    S,
    row,
    part,
    S0: gl.constexpr,
    K0: gl.constexpr,
    SELECTED: gl.constexpr,
    BLOCK: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    tokens = part * BLOCK + gl.arange(0, BLOCK, gl.SliceLayout(1, LAYOUT))
    if SELECTED % BLOCK == 0:
        slots = gl.load(S + row * S0 + tokens).to(gl.int64)
        active = slots >= 0
    else:
        slots = gl.load(S + row * S0 + tokens, tokens < SELECTED, -1).to(gl.int64)
        active = (tokens < SELECTED) & (slots >= 0)
    dims = gl.arange(0, 512, gl.SliceLayout(0, LAYOUT))
    return (slots[:, None] * K0 + dims[None, :], active)


@gluon.jit
def _load_selection(
    S,
    row,
    part,
    S0: gl.constexpr,
    SELECTED: gl.constexpr,
    BLOCK: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    tokens = part * BLOCK + gl.arange(0, BLOCK, LAYOUT)
    if SELECTED % BLOCK == 0:
        slots = gl.load(S + row * S0 + tokens).to(gl.int64)
        active = slots >= 0
    else:
        slots = gl.load(S + row * S0 + tokens, tokens < SELECTED, -1).to(gl.int64)
        active = (tokens < SELECTED) & (slots >= 0)
    return (slots, active)


@gluon.jit
def _partial(
    Q,
    KV,
    S,
    P,
    ML,
    VALID,
    Q0: gl.constexpr,
    Q1: gl.constexpr,
    K0: gl.constexpr,
    S0: gl.constexpr,
    H: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    SCALE,
    BLOCK: gl.constexpr,
    WIDTH: gl.constexpr,
    BUFFER: gl.constexpr,
    ASYNC: gl.constexpr,
    WARPS: gl.constexpr,
    QW: gl.constexpr,
    PACK: gl.constexpr,
    PAD: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    STORE_BUFFER: gl.constexpr,
    M: gl.constexpr,
    Q_ASYNC: gl.constexpr,
    QK_WIDTH: gl.constexpr,
    ROPE_STAGE: gl.constexpr = 0,
    PV_WIDTH: gl.constexpr = 512,
):
    row = gl.program_id(0) % M
    linear = gl.program_id(0) // M
    part = linear // (512 // WIDTH)
    value_group = linear % (512 // WIDTH)
    cache_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [CACHE_ROWS, 64 // CACHE_ROWS], [WARPS, 1], [1, 0]
    )
    query_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 16], [QW, WARPS // QW], [1, 0]
    )
    matrix_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, WARPS]
    )
    qa_layout: gl.constexpr = gl.DotOperandLayout(0, matrix_layout, 8)
    kb_layout: gl.constexpr = gl.DotOperandLayout(1, matrix_layout, 8)
    compact_heads: gl.constexpr = (
        2 if BLOCK == 128 or (BLOCK == 64 and WIDTH == 128) else 4
    )
    compact_layout: gl.constexpr = gl.BlockedLayout(
        [1, PACK], [compact_heads, 64 // compact_heads], [WARPS, 1], [1, 0]
    )
    value_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=False, warps_per_cta=[1, WARPS]
    )
    pa_layout: gl.constexpr = gl.DotOperandLayout(0, value_layout, 8)
    vb_layout: gl.constexpr = gl.DotOperandLayout(1, value_layout, 8)
    if ASYNC:
        if BLOCK == 128 or (BLOCK == 64 and M > 2):
            copy_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [32, 0]]
                + ([[64, 0]] if BLOCK == 128 else []),
                lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
                warp_bases=[[4, 0], [8, 0]],
                block_bases=[],
                shape=[BLOCK, 512],
            )
            shared_layout: gl.constexpr = gl.PaddedSharedLayout(
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
                    [32, 0],
                ]
                + ([[64, 0]] if BLOCK == 128 else []),
                [],
                [BLOCK, 512],
            )
        else:
            copy_layout: gl.constexpr = gl.BlockedLayout(
                [1, 16], [2, 32], [WARPS, 1], [1, 0]
            )
            shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                [[1024, 16]], [BLOCK, 512], [1, 0]
            )
        if M > 2:
            offsets, copy_active = _copy_offsets(
                S, row, part, S0, K0, SELECTED, BLOCK, copy_layout
            )
    if ASYNC and M <= 8:
        validity_layout: gl.constexpr = gl.SliceLayout(0, compact_layout)
    else:
        validity_layout: gl.constexpr = gl.SliceLayout(1, cache_layout)
    if M == 2:
        slots, active = _load_selection(
            S, row, part, S0, SELECTED, BLOCK, validity_layout
        )
    probability_shared = gl.allocate_shared_memory(
        gl.bfloat16,
        (16, BLOCK),
        gl.PaddedSharedLayout.with_identity_for([[BLOCK, 8]], [16, BLOCK], [1, 0]),
    )
    if ROPE_STAGE:
        rope_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8], [8, 8], [WARPS, 1], [1, 0]
        )
        rope_shared = gl.allocate_shared_memory(
            gl.bfloat16,
            (8, 64),
            gl.PaddedSharedLayout.with_identity_for([[512, 16]], [8, 64], [1, 0]),
        )
        rope_copy_heads = gl.arange(0, 8, gl.SliceLayout(1, rope_copy_layout))
        rope_copy_dims = 512 + gl.arange(0, 64, gl.SliceLayout(0, rope_copy_layout))
        if ROPE_STAGE == 1:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                rope_shared,
                Q,
                (row * Q0 + rope_copy_heads[:, None] * Q1 + rope_copy_dims[None, :]).to(
                    gl.int32
                ),
                mask=rope_copy_heads[:, None] < H,
            )
    heads = gl.arange(0, 8, gl.SliceLayout(1, query_layout))
    q_dims = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    if Q_ASYNC:
        query_shared = gl.allocate_shared_memory(
            gl.bfloat16,
            (8, 512),
            gl.PaddedSharedLayout.with_identity_for([[512, 16]], [8, 512], [1, 0]),
        )
        query_copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 8], [1, 64], [WARPS, 1], [1, 0]
        )
        copy_heads = gl.arange(0, 8, gl.SliceLayout(1, query_copy_layout))
        copy_qdims = gl.arange(0, 512, gl.SliceLayout(0, query_copy_layout))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            query_shared,
            Q,
            (row * Q0 + copy_heads[:, None] * Q1 + copy_qdims[None, :]).to(gl.int32),
            mask=copy_heads[:, None] < H,
        )
    else:
        query = gl.load(
            Q + row * Q0 + heads[:, None] * Q1 + q_dims[None, :],
            heads[:, None] < H,
            0.0,
        )
        query_shared = gl.allocate_shared_memory(
            gl.bfloat16,
            (8, 512),
            gl.PaddedSharedLayout.with_identity_for([[512, 16]], [8, 512], [1, 0]),
            query,
        )
    if M != 2:
        slots, active = _load_selection(
            S, row, part, S0, SELECTED, BLOCK, validity_layout
        )
    dims = gl.arange(0, 512, gl.SliceLayout(0, cache_layout))
    if ASYNC:
        latent_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty, (BLOCK, 512), shared_layout
        )
        if M <= 2:
            offsets, copy_active = _copy_offsets(
                S, row, part, S0, K0, SELECTED, BLOCK, copy_layout
            )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            latent_shared, KV, offsets.to(gl.int32), mask=copy_active[:, None]
        )
    else:
        latent = _cache_load(
            KV, slots[:, None] * K0 + dims[None, :], active[:, None], BUFFER
        )
        latent_shared = gl.allocate_shared_memory(
            KV.dtype.element_ty,
            (BLOCK, 512),
            gl.PaddedSharedLayout.with_identity_for([[512, PAD]], [BLOCK, 512], [1, 0]),
            latent,
        )
    q_rope_dims = 512 + gl.arange(0, 64, gl.SliceLayout(0, qa_layout))
    rope_heads = gl.arange(0, 8, gl.SliceLayout(1, qa_layout))
    if ROPE_STAGE == 2:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            rope_shared,
            Q,
            (row * Q0 + rope_copy_heads[:, None] * Q1 + rope_copy_dims[None, :]).to(
                gl.int32
            ),
            mask=rope_copy_heads[:, None] < H,
        )
    if ASYNC or Q_ASYNC or ROPE_STAGE:
        gl.amd.cdna4.async_copy.commit_group()
    if not ROPE_STAGE:
        q_rope = gl.load(
            Q + row * Q0 + rope_heads[:, None] * Q1 + q_rope_dims[None, :],
            rope_heads[:, None] < H,
            0.0,
        )
    rope_tokens = part * BLOCK + gl.arange(0, BLOCK, gl.SliceLayout(0, kb_layout))
    rope_slots = gl.load(S + row * S0 + rope_tokens, rope_tokens < SELECTED, -1).to(
        gl.int64
    )
    rope_dims = 512 + gl.arange(0, 64, gl.SliceLayout(1, kb_layout))
    key_rope = _cache_load(
        KV,
        rope_slots[None, :] * K0 + rope_dims[:, None],
        (rope_slots >= 0)[None, :],
        BUFFER,
    )
    if ASYNC or Q_ASYNC or ROPE_STAGE:
        gl.amd.cdna4.async_copy.wait_group(0)
    if ROPE_STAGE:
        q_rope = rope_shared.load(qa_layout)
    if M == 1:
        if value_group == 0:
            value_view = latent_shared.slice(0, 128, 1)
        elif value_group == 1:
            value_view = latent_shared.slice(128, 128, 1)
        elif value_group == 2:
            value_view = latent_shared.slice(256, 128, 1)
        else:
            value_view = latent_shared.slice(384, 128, 1)
    scores = gl.zeros((8, BLOCK), gl.float32, matrix_layout)
    for k in gl.static_range(512 // QK_WIDTH):
        q = query_shared.slice(k * QK_WIDTH, QK_WIDTH, 1).load(qa_layout)
        key = (
            latent_shared.slice(k * QK_WIDTH, QK_WIDTH, 1)
            .permute((1, 0))
            .load(kb_layout)
        )
        scores = gl.amd.cdna4.mfma(q, key.to(gl.bfloat16), scores)
    scores = gl.amd.cdna4.mfma(q_rope, key_rope.to(gl.bfloat16), scores)
    scores = gl.convert_layout(scores, compact_layout) * SCALE
    mask = gl.convert_layout(active, gl.SliceLayout(0, compact_layout))
    scores = gl.where(mask[None, :], scores, -float("inf"))
    maximum = gl.max(scores, 1)
    any_active = gl.sum(mask.to(gl.int32), 0) > 0
    safe_max = gl.where(any_active, maximum, 0.0)
    probability = gl.where(mask[None, :], _exp(scores - safe_max[:, None]), 0.0)
    denominator = gl.sum(probability, 1)
    high = probability.to(gl.bfloat16)
    residual = (probability - high.to(gl.float32)).to(gl.bfloat16)
    paired = gl.reshape(gl.permute(gl.join(high, residual), (0, 2, 1)), (16, BLOCK))
    probability_shared.store(paired)
    if PV_WIDTH < WIDTH:
        gl.static_assert(WIDTH == 512 and M > 4)
        prob = probability_shared.load(pa_layout)
        for fragment in gl.static_range(WIDTH // PV_WIDTH):
            value = (
                latent_shared.slice(fragment * PV_WIDTH, PV_WIDTH, 1)
                .load(vb_layout)
                .to(gl.bfloat16)
            )
            products = gl.amd.cdna4.mfma(
                prob, value, gl.zeros((16, PV_WIDTH), gl.float32, value_layout)
            )
            upper, lower = gl.split(
                gl.permute(gl.reshape(products, (8, 2, PV_WIDTH)), (0, 2, 1))
            )
            fragment_numerator = upper + lower
            _store_fragment(
                P,
                fragment_numerator,
                row,
                part,
                H,
                SPLITS,
                fragment,
                PV_WIDTH,
                STORE_BUFFER,
            )
    else:
        if M == 1:
            value = value_view.load(vb_layout).to(gl.bfloat16)
        else:
            value = _value_fragment(
                latent_shared, value_group, WIDTH, vb_layout, M == 2
            )
        products = gl.amd.cdna4.mfma(
            probability_shared.load(pa_layout),
            value,
            gl.zeros((16, WIDTH), gl.float32, value_layout),
        )
        upper, lower = gl.split(
            gl.permute(gl.reshape(products, (8, 2, WIDTH)), (0, 2, 1))
        )
        numerator = upper + lower
    if PV_WIDTH >= WIDTH:
        out_layout: gl.constexpr = numerator.type.layout
        out_heads = gl.arange(0, 8, gl.SliceLayout(1, out_layout))
        out_dims = value_group * WIDTH + gl.arange(
            0, WIDTH, gl.SliceLayout(0, out_layout)
        )
        if M == 2:
            p_offset = (
                (row * H + out_heads[:, None]) * SPLITS + part
            ) * 640 + out_dims[None, :]
        elif M > 4:
            p_offset = (
                (row * H + out_heads[:, None]) * SPLITS + part
            ) * 528 + out_dims[None, :]
        else:
            p_offset = (
                (row * SPLITS + part) * H + out_heads[:, None]
            ) * 512 + out_dims[None, :]
        if STORE_BUFFER:
            gl.amd.cdna4.buffer_store(
                numerator, P, p_offset.to(gl.int32), out_heads[:, None] < H
            )
        else:
            gl.store(P + p_offset, numerator, out_heads[:, None] < H)
    if value_group == 0:
        stat_heads = gl.arange(0, 8, gl.SliceLayout(1, compact_layout))
        if M > 4:
            stat_record = ((row * H + stat_heads) * SPLITS + part) * 528 + 512
            if M > 8 and STORE_BUFFER:
                gl.amd.cdna4.buffer_store(
                    maximum, P, stat_record.to(gl.int32), stat_heads < H
                )
                gl.amd.cdna4.buffer_store(
                    denominator, P, (stat_record + 1).to(gl.int32), stat_heads < H
                )
                gl.amd.cdna4.buffer_store(
                    gl.full((8,), 1, gl.int32, gl.SliceLayout(1, compact_layout))
                    * any_active.to(gl.int32),
                    P.to(gl.pointer_type(gl.int32)),
                    (stat_record + 2).to(gl.int32),
                    stat_heads < H,
                )
            else:
                gl.store(P + stat_record, maximum, stat_heads < H)
                gl.store(P + stat_record + 1, denominator, stat_heads < H)
                gl.store(
                    P.to(gl.pointer_type(gl.int32)) + stat_record + 2,
                    gl.full((8,), 1, gl.int32, gl.SliceLayout(1, compact_layout))
                    * any_active.to(gl.int32),
                    stat_heads < H,
                )
        else:
            stat_record = ((row * H + stat_heads) * SPLITS + part) * 2
            gl.store(ML + stat_record, maximum, stat_heads < H)
            gl.store(ML + stat_record + 1, denominator, stat_heads < H)
            gl.store(VALID + row * SPLITS + part, any_active.to(gl.int32))


@gluon.jit
def _merge(
    P,
    ML,
    VALID,
    O,
    H: gl.constexpr,
    SPLITS: gl.constexpr,
    REDUCE: gl.constexpr,
    WIDTH: gl.constexpr,
    WARPS: gl.constexpr,
    STORE_BUFFER: gl.constexpr,
    M: gl.constexpr,
    VALUE_WIDTH: gl.constexpr,
):
    row = gl.program_id(0) % M
    value_group = gl.program_id(0) // M % (512 // VALUE_WIDTH)
    rest = gl.program_id(0) // (M * (512 // VALUE_WIDTH))
    head = rest % H
    chunk = value_group * (VALUE_WIDTH // WIDTH) + rest // H
    row_head = row * H + head
    if WIDTH == 16:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, WARPS], [1, 0])
    elif WIDTH == 32 or WIDTH == 64 or WIDTH == 128:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, WARPS], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, WIDTH // (16 * WARPS)], [4, 16], [1, WARPS], [1, 0]
        )
    r = gl.arange(0, REDUCE, gl.SliceLayout(1, layout))
    d = chunk * WIDTH + gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    if M > 4:
        stat_record = (row_head * SPLITS + r) * 528 + 512
        maximum = gl.load(P + stat_record, r < SPLITS, -float("inf"))
        denominator = gl.load(P + stat_record + 1, r < SPLITS, 0.0)
        active = (
            gl.load(P.to(gl.pointer_type(gl.int32)) + stat_record + 2, r < SPLITS, 0)
            != 0
        )
    else:
        maximum = gl.load(ML + (row_head * SPLITS + r) * 2, r < SPLITS, -float("inf"))
        denominator = gl.load(ML + (row_head * SPLITS + r) * 2 + 1, r < SPLITS, 0.0)
        active = gl.load(VALID + row_head // H * SPLITS + r, r < SPLITS, 0) != 0
    global_max = gl.max(maximum, 0)
    factor = gl.where(active, _exp(maximum - global_max), 0.0)
    if M == 2:
        p_offset = (row_head * SPLITS + r[:, None]) * 640 + d[None, :]
    elif M > 4:
        p_offset = (row_head * SPLITS + r[:, None]) * 528 + d[None, :]
    else:
        p_offset = ((row_head // H * SPLITS + r[:, None]) * H + row_head % H) * 512 + d[
            None, :
        ]
    values = _cache_load(P, p_offset, r[:, None] < SPLITS, STORE_BUFFER)
    numerator = gl.sum(values * factor[:, None], 0)
    total = gl.sum(denominator * factor, 0)
    inverse = 1.0 / gl.where(total > 0, total, 1.0)
    result = numerator * inverse
    gl.store(O + row_head * 512 + d, result.to(gl.bfloat16))


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    m, h, d = query.shape
    assert m > 0 and h == 8 and (d == 576) and (query.dtype == torch.bfloat16)
    assert kv_cache.ndim == 2 and kv_cache.shape[1] == 576
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert selected_slots.ndim == 2 and selected_slots.shape[0] == m
    assert 0 < selected_slots.shape[1] <= 2048
    assert selected_slots.dtype in (torch.int32, torch.int64)
    assert query.stride(-1) == kv_cache.stride(-1) == selected_slots.stride(-1) == 1
    assert (
        query.device == kv_cache.device == selected_slots.device and softmax_scale > 0
    )
    block = 32 if m == 1 else 64 if m <= 8 else 128
    width = 128 if m <= 2 else 256 if m <= 4 else 512
    splits = triton.cdiv(selected_slots.shape[1], block)
    producer_warps = 2 if m == 1 else 4
    query_waves = 1 if m <= 4 else 2
    score_pack = 2 if m <= 2 else 4
    cache_rows = 8 if m == 1 else 4 if m == 2 else 16
    cache_pad = 8 if m == 1 else 16
    merge_width = (
        16 if m == 1 else 32 if m == 2 else 64 if m <= 4 else 128 if m <= 8 else 256
    )
    merge_warps = 1 if m <= 2 else 2 if m <= 4 else 4
    output = torch.empty((m, h, 512), device=query.device, dtype=torch.bfloat16)
    if m > 4:
        partial = torch.empty(
            (m, h, splits, 528), device=query.device, dtype=torch.float32
        )
        stats = partial
        valid = partial.view(torch.int32)
    else:
        partial = torch.empty(
            (m, splits, h, 640 if m == 2 else 512),
            device=query.device,
            dtype=torch.float32,
        )
        stats = torch.empty((m, h, splits, 2), device=query.device, dtype=torch.float32)
        valid = torch.empty((m, splits), device=query.device, dtype=torch.int32)
    extent = (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576
    use_buffer = extent < 2**31
    use_async = (
        use_buffer
        and kv_cache.stride(0) % 16 == 0
        and (kv_cache.storage_offset() % 16 == 0)
    )
    query_extent = ((m - 1) * query.stride(0) + (h - 1) * query.stride(1) + 576) * 2
    query_buffer = query_extent < 2**31
    query_async = (
        use_async
        and query_buffer
        and (query.stride(0) % 8 == 0)
        and (query.stride(1) % 8 == 0)
        and (query.storage_offset() % 8 == 0)
    )
    partial_buffer = m >= 3 and partial.numel() * 4 < 2**31
    _partial[m * splits * (512 // width),](
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
        softmax_scale,
        block,
        width,
        use_buffer,
        use_async,
        producer_warps,
        query_waves,
        score_pack,
        cache_pad,
        cache_rows,
        partial_buffer,
        m,
        query_async,
        128 if m > 8 else 512,
        ROPE_STAGE=(1 if m <= 2 else 2) if query_async else 0,
        PV_WIDTH=256 if 4 < m <= 8 else 512,
        num_warps=producer_warps,
        enable_fp_fusion=False,
        allow_flush_denorm=True,
        llvm_fn_attrs=[["amdgpu-sched-strategy", "iterative-ilp"]]
        if use_async and (2 < m <= 4 or m > 8)
        else [],
    )
    _merge[m * h * (512 // merge_width),](
        partial,
        stats,
        valid,
        output,
        h,
        splits,
        triton.next_power_of_2(splits),
        merge_width,
        merge_warps,
        False if m > 4 else partial_buffer,
        m,
        width,
        num_warps=merge_warps,
        enable_fp_fusion=False,
        allow_flush_denorm=True,
    )
    return output
