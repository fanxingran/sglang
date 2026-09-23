# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m256.py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
)
_SOFTMAX = gl.constexpr(gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0]))
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


@gluon.jit
def _load_inputs(
    Q,
    Slots,
    row,
    part,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    S_ROW: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK: gl.constexpr,
):
    witnesses = ()
    if SPLITS == 4:
        for tile in gl.static_range(CHUNK):
            witnesses += (
                gl.load(Slots + row * S_ROW + (tile * SPLITS + part) * BLOCK_N),
            )
    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 2], [1, 0])
    h = gl.arange(0, 16, gl.SliceLayout(1, load_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, load_layout))
    r = gl.arange(0, 64, gl.SliceLayout(0, load_layout))
    q = gl.load(
        Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < HEADS, 0.0
    )
    qr = gl.load(
        Q + row * Q_ROW + h[:, None] * Q_HEAD + 512 + r[None, :],
        h[:, None] < HEADS,
        0.0,
    )
    if SPLITS == 2:
        for tile in gl.static_range(CHUNK):
            witnesses += (
                gl.load(Slots + row * S_ROW + (tile * SPLITS + part) * BLOCK_N),
            )
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, _NARROW_COPY))
    slots = ()
    for tile in gl.static_range(CHUNK):
        slots += (gl.load(Slots + row * S_ROW + (tile * SPLITS + part) * BLOCK_N + n),)
    activity = _scan_slot_chunk(slots, witnesses, CHUNK)
    return (q, qr, slots, activity)


@gluon.jit
def _issue_slot_chunk(
    Slots,
    row,
    part,
    S_ROW: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    CHUNK: gl.constexpr,
    START: gl.constexpr,
):
    witnesses = ()
    slots = ()
    for tile in gl.static_range(CHUNK):
        witnesses += (
            gl.load(Slots + row * S_ROW + ((START + tile) * SPLITS + part) * BLOCK_N),
        )
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, _NARROW_COPY))
    for tile in gl.static_range(CHUNK):
        slots += (
            gl.load(
                Slots + row * S_ROW + ((START + tile) * SPLITS + part) * BLOCK_N + n
            ),
        )
    return (slots, witnesses)


@gluon.jit
def _scan_slot_chunk(slots, witnesses, CHUNK: gl.constexpr):
    activity = ()
    for tile in gl.static_range(CHUNK):
        active = witnesses[tile] >= 0
        if not active:
            active = gl.max(slots[tile], 0) >= 0
        activity += (active,)
    return activity


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
    gl.static_assert(KV_ROW % 16 == 0 and (KV_ROWS - 1) * KV_ROW + 575 < 4294967296)
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
        flat, KV, offsets, slots[:, None] >= 0, cache_modifier=".cv"
    )


@gluon.jit
def _qk_panel(query, key_smem, score):
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, score.type.layout, 16)
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
def _initial_state():
    numerators = ()
    for fragment in gl.static_range(4):
        numerators += (gl.zeros((16, 128), gl.float32, _MMA),)
    maximum = gl.full((16,), -float("inf"), gl.float32, gl.SliceLayout(1, _SOFTMAX))
    denominator = gl.zeros((16,), gl.float32, gl.SliceLayout(1, _SOFTMAX))
    return (numerators, maximum, denominator)


@gluon.jit
def _start_pipeline(
    KV,
    q,
    qr,
    first_slots,
    first_active,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    key0 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    key1 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    if first_active:
        _prefetch_panel(key0, KV, first_slots, KV_ROW, KV_ROWS, 0, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
        _prefetch_panel(key1, KV, first_slots, KV_ROW, KV_ROWS, 128, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
    query_smem = gl.allocate_shared_memory(gl.bfloat16, [16, 512], _QUERY_SHARED, q)
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], _QUERY_SHARED, qr
    )
    query_panels = _read_query_panels(query_smem, rotary_query_smem)
    numerators, maximum, denominator = _initial_state()
    rotary_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    key2 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    key3 = gl.allocate_shared_memory(KV.dtype.element_ty, [BLOCK_N, 128], shared_layout)
    rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_layout
    )
    other_key0 = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    other_rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_layout
    )
    buffers = (
        (key0, key1, key2, key3, rotary),
        (other_key0, key1, key2, key3, other_rotary),
    )
    if first_active:
        for panel in gl.static_range(2, 5):
            _prefetch_panel(
                buffers[0][panel],
                KV,
                first_slots,
                KV_ROW,
                KV_ROWS,
                panel * 128,
                BLOCK_N,
            )
            gl.amd.cdna4.async_copy.commit_group()
    return (buffers, query_panels, numerators, maximum, denominator)


@gluon.jit
def _wait_latent(PANEL: gl.constexpr):
    waits: gl.constexpr = (
        "s_waitcnt vmcnt(7); v_mov_b32 $0, 0;",
        "s_waitcnt vmcnt(5); v_mov_b32 $0, 0;",
        "s_waitcnt vmcnt(3); v_mov_b32 $0, 0;",
        "s_waitcnt vmcnt(1); v_mov_b32 $0, 0;",
    )
    gl.inline_asm_elementwise(
        waits[PANEL],
        constraints="=v,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _qk_tile(query_panels, current, BLOCK_N: gl.constexpr):
    score = gl.zeros((16, BLOCK_N), gl.float32, _MMA)
    for panel in gl.static_range(5):
        if panel < 4:
            _wait_latent(panel)
        else:
            gl.amd.cdna4.async_copy.wait_group(0)
        score = _qk_panel(query_panels[panel], current[panel], score)
    return score


@gluon.jit
def _accumulate_tile(
    KV,
    current,
    valid,
    successor,
    successor_slots,
    successor_active,
    score,
    numerators,
    maximum,
    denominator,
    scale,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    INITIAL: gl.constexpr,
    HAS_SUCCESSOR: gl.constexpr,
):
    rhs: gl.constexpr = gl.DotOperandLayout(1, _MMA, 8)
    if HAS_SUCCESSOR:
        if successor_active:
            _prefetch_panel(
                successor[0], KV, successor_slots, KV_ROW, KV_ROWS, 0, BLOCK_N
            )
            gl.amd.cdna4.async_copy.commit_group()
    p, alpha, maximum, denominator = _online_softmax(
        score, valid, maximum, denominator, current[4], scale, BLOCK_N, INITIAL
    )
    next_numerators = ()
    for pv_panel in gl.static_range(4):
        value = gl.amd.cdna4.async_copy.load_shared_relaxed(
            current[pv_panel], _PV_OPERAND
        ).to(gl.bfloat16)
        if HAS_SUCCESSOR and pv_panel > 0:
            if successor_active:
                gl.barrier()
                _prefetch_panel(
                    successor[pv_panel],
                    KV,
                    successor_slots,
                    KV_ROW,
                    KV_ROWS,
                    pv_panel * 128,
                    BLOCK_N,
                )
                gl.amd.cdna4.async_copy.commit_group()
                if pv_panel == 3:
                    _prefetch_panel(
                        successor[4], KV, successor_slots, KV_ROW, KV_ROWS, 512, BLOCK_N
                    )
                    gl.amd.cdna4.async_copy.commit_group()
        numerator = gl.amd.cdna4.mfma(
            p, gl.convert_layout(value, rhs), numerators[pv_panel] * alpha[:, None]
        )
        next_numerators += (numerator,)
    return (next_numerators, maximum, denominator)


@gluon.jit
def _restart_after_empty(
    KV,
    successor,
    successor_slots,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    gl.barrier()
    _prefetch_panel(successor[0], KV, successor_slots, KV_ROW, KV_ROWS, 0, BLOCK_N)
    gl.amd.cdna4.async_copy.commit_group()
    for panel in gl.static_range(1, 4):
        _prefetch_panel(
            successor[panel], KV, successor_slots, KV_ROW, KV_ROWS, panel * 128, BLOCK_N
        )
        gl.amd.cdna4.async_copy.commit_group()
    _prefetch_panel(successor[4], KV, successor_slots, KV_ROW, KV_ROWS, 512, BLOCK_N)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _pipeline_short(
    KV,
    q,
    qr,
    slots,
    activity,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    scale,
    BLOCK_N: gl.constexpr,
    TILES: gl.constexpr,
    Slots,
    row,
    part,
    S_ROW: gl.constexpr,
    SPLITS: gl.constexpr,
    CHUNK: gl.constexpr,
):
    buffers, query_panels, numerators, maximum, denominator = _start_pipeline(
        KV, q, qr, slots[0], activity[0], KV_ROW, KV_ROWS, BLOCK_N
    )
    for step in gl.static_range(TILES):
        current = buffers[step % 2]
        successor = buffers[(step + 1) % 2]
        current_slots = slots[step % CHUNK]
        current_active = activity[step % CHUNK]
        if step + 1 < TILES:
            if (step + 1) % CHUNK == 0:
                slots, witnesses = _issue_slot_chunk(
                    Slots, row, part, S_ROW, SPLITS, BLOCK_N, CHUNK, step + 1
                )
                activity = _scan_slot_chunk(slots, witnesses, CHUNK)
            successor_slots = slots[(step + 1) % CHUNK]
            successor_active = activity[(step + 1) % CHUNK]
        if current_active:
            valid = gl.convert_layout(current_slots >= 0, gl.SliceLayout(0, _SOFTMAX))
            score = _qk_tile(query_panels, current, BLOCK_N)
            numerators, maximum, denominator = _accumulate_tile(
                KV,
                current,
                valid,
                successor,
                successor_slots,
                successor_active,
                score,
                numerators,
                maximum,
                denominator,
                scale,
                KV_ROW,
                KV_ROWS,
                BLOCK_N,
                step == 0,
                step + 1 < TILES,
            )
        if step + 1 < TILES:
            if not current_active and successor_active:
                _restart_after_empty(
                    KV, successor, successor_slots, KV_ROW, KV_ROWS, BLOCK_N
                )
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
    CHUNK: gl.constexpr,
    ROWS: gl.constexpr,
):
    row, part = (gl.program_id(0), gl.program_id(1))
    if ROWS % 4 == 0:
        linear = row + part * ROWS
        row = linear // 16 * 4 + linear % 4
        part = linear // 4 % 4
    q, qr, slots, activity = _load_inputs(
        Q, Slots, row, part, Q_ROW, Q_HEAD, S_ROW, HEADS, SPLITS, BLOCK_N, CHUNK
    )
    h = gl.arange(0, 16, gl.SliceLayout(1, _MMA))
    gl.static_assert(SPLITS == 4)
    numerators, maximum, denominator = _pipeline_short(
        KV,
        q,
        qr,
        slots,
        activity,
        KV_ROW,
        KV_ROWS,
        scale,
        BLOCK_N,
        TILES,
        Slots,
        row,
        part,
        S_ROW,
        SPLITS,
        CHUNK,
    )
    for panel in gl.static_range(4):
        _store_partial(
            numerators[panel], Partial, row, part, h[:, None], panel, HEADS, SPLITS
        )
    store_layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
    statistics = gl.convert_layout(
        gl.join(maximum, denominator), store_layout, assert_trivial=True
    )
    sh = gl.arange(0, 16, gl.SliceLayout(1, store_layout))
    field = gl.arange(0, 2, gl.SliceLayout(0, store_layout))
    offsets = sh[:, None] * 2 + field[None, :]
    gl.amd.cdna4.buffer_store(
        statistics,
        Stats + (row * SPLITS + part) * HEADS * 2,
        offsets,
        sh[:, None] < HEADS,
        cache=".wt",
    )


@gluon.jit
def _merge_statistics(
    Stats,
    row,
    h,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    OUTPUT_LAYOUT: gl.constexpr,
):
    stats_layout: gl.constexpr = h.type.layout.parent
    split = gl.arange(0, SPLITS, gl.SliceLayout(0, stats_layout))
    maximum = gl.load(
        Stats + ((row * SPLITS + split[None, :]) * HEADS + h[:, None]) * 2
    )
    den = gl.load(
        Stats + ((row * SPLITS + split[None, :]) * HEADS + h[:, None]) * 2 + 1
    )
    global_max = gl.max(maximum, 1)
    global_max = gl.where(global_max == -float("inf"), 0.0, global_max)
    weights = gl.exp2((maximum - global_max[:, None]) * _LOG2_E)
    denominator = gl.convert_layout(
        gl.sum(den * weights, 1), gl.SliceLayout(1, OUTPUT_LAYOUT)
    )
    return (weights, denominator)


@gluon.jit
def _merge_native(Partial, Stats, Out, HEADS: gl.constexpr, SPLITS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [0, 1])
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [2, 1], [0, 1])
    row = gl.program_id(0)
    panel = gl.program_id(1).to(gl.uint32)
    h = gl.arange(0, 8, gl.SliceLayout(1, stats_layout))
    weights, denominator = _merge_statistics(Stats, row, h, HEADS, SPLITS, layout)
    oh = gl.arange(0, 8, gl.SliceLayout(1, layout))
    pd = gl.arange(0, 128, gl.SliceLayout(0, layout))
    partial_base = Partial + (row * 4 * SPLITS + panel * SPLITS) * HEADS * 128
    partial_offset = pd[None, :] // 8 * HEADS * 8 + oh[:, None] * 8 + pd[None, :] % 8
    acc = gl.zeros((8, 128), gl.float32, layout)
    for part in gl.static_range(SPLITS):
        index = gl.full((8, 1), part, gl.int32, stats_layout)
        weight = gl.convert_layout(
            gl.sum(gl.gather(weights, index, 1), 1), gl.SliceLayout(1, layout)
        )
        p = gl.amd.cdna4.buffer_load(
            partial_base, part * HEADS * 128 + partial_offset, cache=".cg"
        ).to(gl.float32)
        acc = gl.fma(p, weight[:, None], acc)
    reciprocal = 1.0 / gl.where(denominator > 0.0, denominator, 1.0)
    result = acc * reciprocal[:, None]
    d = panel * 128 + pd // 8 * 4 + pd % 8 // 4 * 64 + pd % 4
    output_offsets = oh[:, None] * 512 + d[None, :]
    output_offsets = gl.max_contiguous(gl.multiple_of(output_offsets, [1, 4]), [1, 4])
    gl.amd.cdna4.buffer_store(
        result.to(gl.bfloat16), Out + row * HEADS * 512, output_offsets, cache=".wt"
    )


_PAIR_MMA = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 1, 4]
    )
)
_PAIR_SOFTMAX = gl.constexpr(
    gl.BlockedLayout([1, 1, 4], [1, 4, 16], [2, 4, 1], [2, 1, 0])
)
_PAIR_QK = gl.constexpr(gl.BlockedLayout([1, 16, 1], [1, 4, 16], [2, 1, 4], [2, 1, 0]))
_PAIR_PV = gl.constexpr(gl.BlockedLayout([1, 8, 1], [1, 4, 16], [2, 1, 4], [2, 1, 0]))
_PAIR_COPY = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[128, 128],
    )
)


@gluon.jit
def _or_activity(a, b):
    return a | b


@gluon.jit
def _pair_scan_future(slots):
    packed = gl.full((128,), 0, gl.int32, slots[0].type.layout)
    for tile in gl.static_range(8):
        packed = packed | gl.where(slots[tile] >= 0, 1 << tile, 0)
    bits = gl.reduce(packed, 0, _or_activity)
    activity = ()
    for tile in gl.static_range(8):
        activity += (bits & 1 << tile != 0,)
    return activity


@gluon.jit
def _pair_issue_slots(Slots, row, S_ROW: gl.constexpr, START: gl.constexpr):
    witnesses = ()
    slots = ()
    n = gl.arange(0, 128, gl.SliceLayout(1, _PAIR_COPY))
    for tile in gl.static_range(8):
        witnesses += (gl.load(Slots + row * S_ROW + (START + tile) * 128),)
    for tile in gl.static_range(8):
        slots += (gl.load(Slots + row * S_ROW + (START + tile) * 128 + n),)
    return (slots, witnesses)


@gluon.jit
def _pair_prefetch(dest, KV, slots, KV_ROW: gl.constexpr, CHANNEL: gl.constexpr):
    if CHANNEL == 512:
        WIDTH: gl.constexpr = 64
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [8, 1], [1, 0])
    else:
        WIDTH: gl.constexpr = 128
        layout: gl.constexpr = _PAIR_COPY
    slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
    columns = gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    rows = gl.arange(0, 128, gl.SliceLayout(1, layout))
    flat = dest._reinterpret(
        KV.dtype.element_ty, [128, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    physical_columns = columns[None, :] ^ rows[:, None] % (WIDTH // 16) * 16
    columns_offset = CHANNEL + gl.max_contiguous(
        gl.multiple_of(physical_columns, [1, 16]), [1, 16]
    )
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
        flat, KV, offsets, slots[:, None] >= 0, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _pair_softmax(
    score, valid, maximum, denominator, rotary, scale, INITIAL: gl.constexpr
):
    gl.barrier()
    score_smem = rotary._reinterpret(
        gl.float32, [2, 16, 64], gl.SwizzledSharedLayout(4, 1, 16, [2, 1, 0])
    )
    score_smem.store(score)
    score = gl.amd.cdna4.async_copy.load_shared_relaxed(score_smem, _PAIR_SOFTMAX)
    score_smem._keep_alive()
    score = gl.where(valid[:, None, :], score * scale, -float("inf"))
    next_maximum = gl.maximum(maximum, gl.max(score, 2))
    safe_maximum = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
    if not INITIAL:
        alpha = gl.exp2((maximum - safe_maximum) * _LOG2_E)
        alpha_smem = gl.allocate_shared_memory(
            gl.float32, [2, 16], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        alpha_smem.store(alpha)
    probability = gl.where(
        valid[:, None, :], gl.exp2((score - safe_maximum[:, :, None]) * _LOG2_E), 0.0
    )
    if INITIAL:
        denominator = gl.sum(probability, 2)
    else:
        denominator = denominator * alpha + gl.sum(probability, 2)
    probability_smem = rotary._reinterpret(
        gl.bfloat16, [2, 32, 64], gl.SwizzledSharedLayout(8, 1, 8, [2, 1, 0])
    ).slice(0, 16, dim=1)
    probability_smem.store(probability.to(gl.bfloat16))
    p = gl.amd.cdna4.async_copy.load_shared_relaxed(
        probability_smem, gl.DotOperandLayout(0, _PAIR_MMA, 8)
    )
    if INITIAL:
        alpha_mma = gl.zeros((2, 16), gl.float32, gl.SliceLayout(2, _PAIR_MMA))
    else:
        alpha_mma = gl.amd.cdna4.async_copy.load_shared_relaxed(
            alpha_smem, gl.SliceLayout(2, _PAIR_MMA)
        )
        alpha_smem._keep_alive()
    probability_smem._keep_alive()
    return (p, alpha_mma, next_maximum, denominator)


@gluon.jit
def _pair_merge(numerators, maximum, denominator, Out, row):
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [2, 1, 0])
    partial_smem = gl.allocate_shared_memory(gl.bfloat16, [2, 16, 512], shared_layout)
    for panel in gl.static_range(4):
        partial_smem.slice(panel * 128, 128, dim=2).store(
            numerators[panel].to(gl.bfloat16)
        )
    stat_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    max_smem = gl.allocate_shared_memory(gl.float32, [2, 16], stat_layout, maximum)
    den_smem = gl.allocate_shared_memory(gl.float32, [2, 16], stat_layout, denominator)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 8, 8], [1, 1, 8], [2, 1, 0])
    maxima = max_smem.slice(0, 8, dim=1).load(gl.SliceLayout(2, layout))
    den = den_smem.slice(0, 8, dim=1).load(gl.SliceLayout(2, layout))
    global_maximum = gl.max(maxima, 0)
    global_maximum = gl.where(global_maximum == -float("inf"), 0.0, global_maximum)
    weights = gl.exp2((maxima - global_maximum[None, :]) * _LOG2_E)
    partial = partial_smem.slice(0, 8, dim=1).load(layout).to(gl.float32)
    p0, p1 = gl.split(partial.permute((1, 2, 0)))
    w0, w1 = gl.split(weights.permute((1, 0)))
    d0, d1 = gl.split(den.permute((1, 0)))
    out_layout: gl.constexpr = p0.type.layout
    w0 = gl.convert_layout(w0, gl.SliceLayout(1, out_layout))
    w1 = gl.convert_layout(w1, gl.SliceLayout(1, out_layout))
    d0 = gl.convert_layout(d0, gl.SliceLayout(1, out_layout))
    d1 = gl.convert_layout(d1, gl.SliceLayout(1, out_layout))
    acc = gl.fma(p0, w0[:, None], gl.zeros((8, 512), gl.float32, out_layout))
    acc = gl.fma(p1, w1[:, None], acc)
    h = gl.arange(0, 8, gl.SliceLayout(1, out_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, out_layout))
    denominator0 = gl.fma(d0, w0, d1 * w1)
    denominator1 = gl.fma(d1, w1, d0 * w0)
    total_den = gl.where(
        d[None, :] & 4 == 0, denominator0[:, None], denominator1[:, None]
    )
    reciprocal = 1.0 / gl.where(total_den > 0.0, total_den, 1.0)
    result = (acc * reciprocal).to(gl.bfloat16)
    offsets = h[:, None] * 512 + d[None, :]
    offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 4]), [1, 4])
    gl.amd.cdna4.buffer_store(result, Out + row * 8 * 512, offsets, cache=".wt")


@gluon.jit
def _fused_attention(
    Q,
    KV,
    Slots,
    Out,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    KV_ROW: gl.constexpr,
    KV_ROWS: gl.constexpr,
    S_ROW: gl.constexpr,
    scale,
):
    gl.static_assert(KV_ROWS <= 16777216 and KV_ROW > 0 and (KV_ROW < 16777216))
    gl.static_assert(KV_ROW % 16 == 0 and (KV_ROWS - 1) * KV_ROW + 575 < 4294967296)
    row = gl.program_id(0)
    q_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 1, 8], [1, 4, 16], [2, 4, 1], [2, 1, 0])
    )
    h = gl.arange(0, 16, gl.SliceLayout(1, q_layout))
    d = gl.arange(0, 512, gl.SliceLayout(0, q_layout))
    r = gl.arange(0, 64, gl.SliceLayout(0, q_layout))
    q = gl.load(Q + row * Q_ROW + h[:, None] * Q_HEAD + d[None, :], h[:, None] < 8, 0.0)
    qr = gl.load(
        Q + row * Q_ROW + h[:, None] * Q_HEAD + 512 + r[None, :], h[:, None] < 8, 0.0
    )
    slots, witnesses = _pair_issue_slots(Slots, row, S_ROW, 0)
    first_active = witnesses[0] >= 0
    if not first_active:
        first_active = gl.max(slots[0], 0) >= 0
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0])
    key0 = gl.allocate_shared_memory(KV.dtype.element_ty, [2, 64, 128], shared_layout)
    key1 = gl.allocate_shared_memory(KV.dtype.element_ty, [2, 64, 128], shared_layout)
    if first_active:
        _pair_prefetch(key0, KV, slots[0], KV_ROW, 0)
        _pair_prefetch(key1, KV, slots[0], KV_ROW, 128)
    activity = (first_active,)
    for tile in gl.static_range(1, 8):
        active = witnesses[tile] >= 0
        if not active:
            active = gl.max(slots[tile], 0) >= 0
        activity += (active,)
    query_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [2, 16, 512],
        shared_layout,
        q[None, :, :].broadcast_to((2, 16, 512)),
    )
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [2, 16, 64],
        shared_layout,
        qr[None, :, :].broadcast_to((2, 16, 64)),
    )
    query_panels = ()
    for panel in gl.static_range(4):
        query_panels += (
            query_smem.slice(panel * 128, 128, dim=2).load(
                gl.DotOperandLayout(0, _PAIR_MMA, 16)
            ),
        )
    query_panels += (rotary_query_smem.load(gl.DotOperandLayout(0, _PAIR_MMA, 16)),)
    query_smem._keep_alive()
    rotary_query_smem._keep_alive()
    key2 = gl.allocate_shared_memory(KV.dtype.element_ty, [2, 64, 128], shared_layout)
    key3 = gl.allocate_shared_memory(KV.dtype.element_ty, [2, 64, 128], shared_layout)
    other_key1 = gl.allocate_shared_memory(
        KV.dtype.element_ty, [2, 64, 128], shared_layout
    )
    other_key0 = gl.allocate_shared_memory(
        KV.dtype.element_ty, [2, 64, 128], shared_layout
    )
    rotary_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [2, 1, 0])
    rotary = gl.allocate_shared_memory(KV.dtype.element_ty, [2, 64, 64], rotary_layout)
    other_rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [2, 64, 64], rotary_layout
    )
    buffers = (
        (key0, key1, key2, key3, rotary),
        (other_key0, other_key1, key2, key3, other_rotary),
    )
    if first_active:
        for panel in gl.static_range(2, 5):
            _pair_prefetch(buffers[0][panel], KV, slots[0], KV_ROW, panel * 128)
    numerators = ()
    for panel in gl.static_range(4):
        numerators += (gl.zeros((2, 16, 128), gl.float32, _PAIR_MMA),)
    maximum = gl.full(
        (2, 16), -float("inf"), gl.float32, gl.SliceLayout(2, _PAIR_SOFTMAX)
    )
    denominator = gl.zeros((2, 16), gl.float32, gl.SliceLayout(2, _PAIR_SOFTMAX))
    current_slots = slots[0]
    current_active = first_active
    for step in gl.static_range(16):
        current = buffers[step % 2]
        successor = buffers[(step + 1) % 2]
        score = gl.zeros((2, 16, 64), gl.float32, _PAIR_MMA)
        valid = gl.full((2, 64), False, gl.int1, gl.SliceLayout(1, _PAIR_SOFTMAX))
        if current_active:
            valid = gl.convert_layout(
                (current_slots >= 0).reshape((2, 64)), gl.SliceLayout(1, _PAIR_SOFTMAX)
            )
            for panel in gl.static_range(5):
                if panel < 4:
                    _wait_latent(panel)
                else:
                    gl.inline_asm_elementwise(
                        "s_waitcnt vmcnt(0); v_mov_b32 $0, 0;",
                        constraints="=v,~{memory}",
                        args=[],
                        dtype=gl.int32,
                        is_pure=False,
                        pack=1,
                    )
                key = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    current[panel].permute((0, 2, 1)), _PAIR_QK
                ).to(gl.bfloat16)
                score = gl.amd.cdna4.mfma(
                    query_panels[panel],
                    gl.convert_layout(key, gl.DotOperandLayout(1, _PAIR_MMA, 16)),
                    score,
                )
        if step + 1 < 16:
            successor_slots = slots[(step + 1) % 8]
            successor_active = activity[(step + 1) % 8]
            if step == 5:
                future_slots, witnesses = _pair_issue_slots(Slots, row, S_ROW, 8)
        if current_active:
            if step + 1 < 16:
                if successor_active:
                    _pair_prefetch(successor[0], KV, successor_slots, KV_ROW, 0)
            p, alpha, maximum, denominator = _pair_softmax(
                score, valid, maximum, denominator, current[4], scale, step == 0
            )
            next_numerators = ()
            for pv_panel in gl.static_range(1, 5):
                value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    current[pv_panel % 4], _PAIR_PV
                ).to(gl.bfloat16)
                if step + 1 < 16 and pv_panel % 4 > 0:
                    if successor_active:
                        if pv_panel != 1:
                            gl.barrier()
                        _pair_prefetch(
                            successor[pv_panel % 4],
                            KV,
                            successor_slots,
                            KV_ROW,
                            pv_panel % 4 * 128,
                        )
                        if pv_panel == 3:
                            _pair_prefetch(
                                successor[4], KV, successor_slots, KV_ROW, 512
                            )
                numerator = gl.amd.cdna4.mfma(
                    p,
                    gl.convert_layout(value, gl.DotOperandLayout(1, _PAIR_MMA, 8)),
                    numerators[pv_panel % 4] * alpha[:, :, None],
                )
                next_numerators += (numerator,)
            numerators = (
                next_numerators[3],
                next_numerators[0],
                next_numerators[1],
                next_numerators[2],
            )
        if step + 1 < 16:
            if not current_active and successor_active:
                gl.barrier()
                for panel in gl.static_range(5):
                    _pair_prefetch(
                        successor[panel], KV, successor_slots, KV_ROW, panel * 128
                    )
            if step == 6:
                slots = future_slots
                activity = _pair_scan_future(slots)
            current_slots = successor_slots
            current_active = successor_active
    _pair_merge(numerators, maximum, denominator, Out, row)


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
    block_n = 64
    splits = 4 if m < 192 else 2
    tiles = selected // (splits * block_n)
    metadata_chunk = tiles // 2
    if splits == 2:
        output = torch.empty((m, heads, 512), device=query.device, dtype=torch.bfloat16)
        _fused_attention[m,](
            query,
            kv_cache,
            selected_slots,
            output,
            query.stride(0),
            query.stride(1),
            kv_cache.stride(0),
            kv_cache.shape[0],
            selected_slots.stride(0),
            softmax_scale,
            num_warps=8,
            num_stages=1,
            waves_per_eu=0,
        )
        return output
    partial = torch.empty(
        (m, 4, splits, 16, heads, 8), device=query.device, dtype=torch.bfloat16
    )
    stats = torch.empty((m, splits, heads, 2), device=query.device, dtype=torch.float32)
    output = torch.empty((m, heads, 512), device=query.device, dtype=torch.bfloat16)
    _partial_attention[m, splits](
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
        metadata_chunk,
        m,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    _merge_native[m, 4](partial, stats, output, heads, splits, num_warps=2)
    return output
