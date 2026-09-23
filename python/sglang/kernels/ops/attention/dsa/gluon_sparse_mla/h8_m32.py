# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m32.py
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _prefetch_panel(
    dest,
    KV,
    slots,
    KV_ROW: gl.constexpr,
    CHANNEL: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PANEL_WIDTH: gl.constexpr = 128,
    USE_U24: gl.constexpr = False,
):
    if CHANNEL == 512:
        WIDTH: gl.constexpr = 64
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
        slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
        CACHE: gl.constexpr = "" if BLOCK_N == 128 else ".cv"
    else:
        WIDTH: gl.constexpr = PANEL_WIDTH
        if BLOCK_N == 64:
            layout: gl.constexpr = gl.DistributedLinearLayout(
                [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
                [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
                [[16, 0], [32, 0]],
                [],
                [64, 128],
            )
        else:
            layout: gl.constexpr = gl.BlockedLayout(
                [1, 16], [1024 // WIDTH, WIDTH // 16], [4, 1], [1, 0]
            )
        slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
        CACHE: gl.constexpr = ".cg" if BLOCK_N == 128 else ".cv"
    columns = gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    rows = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    flat = dest._reinterpret(
        KV.dtype.element_ty, [BLOCK_N, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    physical_columns = columns[None, :] ^ rows[:, None] % (WIDTH // 16) * 16
    if USE_U24:
        offsets = gl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3;",
            constraints="=v,v,s,v",
            args=[slots[:, None].to(gl.uint32), KV_ROW, CHANNEL + physical_columns],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            flat,
            KV,
            gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16]),
            slots[:, None] >= 0,
            cache_modifier=CACHE,
        )
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            flat,
            KV,
            slots[:, None] * KV_ROW
            + CHANNEL
            + gl.max_contiguous(gl.multiple_of(physical_columns, [1, 16]), [1, 16]),
            slots[:, None] >= 0,
            cache_modifier=CACHE,
        )


@gluon.jit
def _qk_panel(query, key_smem, score, BLOCK_N: gl.constexpr, RELAXED: gl.constexpr):
    operand_layout: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0])
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, score.type.layout, 16)
    if BLOCK_N == 64 or RELAXED:
        key = gl.amd.cdna4.async_copy.load_shared_relaxed(
            key_smem.permute((1, 0)), operand_layout
        ).to(gl.bfloat16)
    else:
        key = key_smem.permute((1, 0)).load(operand_layout).to(gl.bfloat16)
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
    query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
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
        alpha = gl.exp2((maximum - safe_maximum) * 1.4426950408889634)
    probability = gl.where(
        valid[None, :],
        gl.exp2((score - safe_maximum[:, None]) * 1.4426950408889634),
        0.0,
    )
    if INITIAL:
        denominator = gl.sum(probability, 1)
    else:
        denominator = denominator * alpha + gl.sum(probability, 1)
    probability_smem = rotary_smem._reinterpret(
        gl.bfloat16, [32, BLOCK_N], query_shared_layout
    ).slice(0, 16, dim=0)
    if not INITIAL:
        alpha_smem = rotary_smem._reinterpret(
            gl.float32, [16 * BLOCK_N], gl.SwizzledSharedLayout(1, 1, 1, [0])
        ).slice(8 * BLOCK_N, 16, dim=0)
        alpha_smem.store(alpha)
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
def _store_partial_octet(
    value,
    Partial,
    row,
    part,
    head,
    CHANNEL: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    CACHE: gl.constexpr = ".cs",
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [0, 1])
    value = value.reshape((16, 2, 16, 4)).permute((0, 2, 1, 3)).reshape((16, 128))
    value = gl.convert_layout(value, layout, assert_trivial=True)
    head = gl.convert_layout(head, layout)
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    base = Partial + ((row * 4 + CHANNEL // 128) * SPLITS + part) * HEADS * 128
    offsets = d[None, :] // 8 * HEADS * 8 + head * 8 + d[None, :] % 8
    gl.amd.cdna4.buffer_store(
        value.to(gl.bfloat16), base, offsets, head < HEADS, cache=CACHE
    )


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
    S_ROW: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    scale,
    BLOCK_N: gl.constexpr,
    USE_U24: gl.constexpr = False,
):
    row = gl.program_id(0)
    part = gl.program_id(1)
    head_group = 0
    witness0 = gl.load(Slots + row * S_ROW + part * BLOCK_N)
    witness1 = gl.load(Slots + row * S_ROW + (SPLITS + part) * BLOCK_N)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    operand_blocked: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 16)
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    h = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, load_layout))
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
    copy_layout: gl.constexpr = gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]],
        [[16, 0], [32, 0]],
        [],
        [64, 128],
    )
    cn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, copy_layout))
    slots0 = gl.load(Slots + row * S_ROW + part * BLOCK_N + cn)
    slots1 = gl.load(Slots + row * S_ROW + (SPLITS + part) * BLOCK_N + cn)
    active0 = witness0 >= 0
    active1 = witness1 >= 0
    if not active0:
        active0 = gl.max(slots0, 0) >= 0
    if not active1:
        active1 = gl.max(slots1, 0) >= 0
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    key0_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    key1_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    if active0:
        _prefetch_panel(key0_smem, KV, slots0, KV_ROW, 0, BLOCK_N, USE_U24=USE_U24)
        gl.amd.cdna4.async_copy.commit_group()
        _prefetch_panel(key1_smem, KV, slots0, KV_ROW, 128, BLOCK_N, USE_U24=USE_U24)
        gl.amd.cdna4.async_copy.commit_group()
    query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 512], query_shared_layout, q
    )
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], query_shared_layout, qr
    )
    query_panels = (
        query_smem.slice(0, 128, dim=1).load(lhs_layout),
        query_smem.slice(128, 128, dim=1).load(lhs_layout),
        query_smem.slice(256, 128, dim=1).load(lhs_layout),
        query_smem.slice(384, 128, dim=1).load(lhs_layout),
        rotary_query_smem.load(lhs_layout),
    )
    query_smem._keep_alive()
    rotary_query_smem._keep_alive()
    numerators = (
        gl.zeros((16, 128), gl.float32, mma_layout),
        gl.zeros((16, 128), gl.float32, mma_layout),
        gl.zeros((16, 128), gl.float32, mma_layout),
        gl.zeros((16, 128), gl.float32, mma_layout),
    )
    oh = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma_layout))
    maximum = gl.full(
        (16,), -float("inf"), gl.float32, gl.SliceLayout(1, softmax_layout)
    )
    denominator = gl.zeros((16,), gl.float32, gl.SliceLayout(1, softmax_layout))
    rotary_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    key2_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    key3_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    rotary_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_shared_layout
    )
    key_panels = (key0_smem, key1_smem, key2_smem, key3_smem, rotary_smem)
    next_key0_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    next_rotary_smem = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_shared_layout
    )
    successor_panels = (
        next_key0_smem,
        key1_smem,
        key2_smem,
        key3_smem,
        next_rotary_smem,
    )
    if active0:
        for panel in gl.static_range(2, 5):
            _prefetch_panel(
                key_panels[panel],
                KV,
                slots0,
                KV_ROW,
                panel * 128,
                BLOCK_N,
                USE_U24=USE_U24,
            )
            gl.amd.cdna4.async_copy.commit_group()
    for step in gl.static_range(2):
        if step == 1:
            tile_panels = successor_panels
            tile_rotary_smem = next_rotary_smem
        else:
            tile_panels = key_panels
            tile_rotary_smem = rotary_smem
        if step == 0:
            slot = slots0
            any_valid = active0
        else:
            slot = slots1
            any_valid = active1
        valid = slot >= 0
        if any_valid:
            gl.amd.cdna4.async_copy.wait_group(4)
            score = gl.zeros((16, BLOCK_N), gl.float32, mma_layout)
            for panel in gl.static_range(5):
                if panel > 0:
                    if step == 0:
                        gl.amd.cdna4.async_copy.wait_group(4 - panel)
                    elif panel < 4:
                        gl.amd.cdna4.async_copy.wait_group(3 - panel)
                score = _qk_panel(
                    query_panels[panel], tile_panels[panel], score, BLOCK_N, False
                )
            if step == 0:
                if active1:
                    _prefetch_panel(
                        next_key0_smem, KV, slots1, KV_ROW, 0, BLOCK_N, USE_U24=USE_U24
                    )
                    gl.amd.cdna4.async_copy.commit_group()
                    _prefetch_panel(
                        next_rotary_smem,
                        KV,
                        slots1,
                        KV_ROW,
                        512,
                        BLOCK_N,
                        USE_U24=USE_U24,
                    )
                    gl.amd.cdna4.async_copy.commit_group()
            valid_softmax = gl.convert_layout(valid, gl.SliceLayout(0, softmax_layout))
            p, alpha_mma, next_maximum, denominator = _online_softmax(
                score,
                valid_softmax,
                maximum,
                denominator,
                tile_rotary_smem,
                scale,
                BLOCK_N,
                step == 0,
            )
            next_numerators = ()
            for panel in gl.static_range(4):
                value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    tile_panels[panel], operand_blocked
                ).to(gl.bfloat16)
                if step == 0:
                    if active1:
                        if panel > 0:
                            gl.barrier()
                            _prefetch_panel(
                                key_panels[panel],
                                KV,
                                slots1,
                                KV_ROW,
                                panel * 128,
                                BLOCK_N,
                                USE_U24=USE_U24,
                            )
                            gl.amd.cdna4.async_copy.commit_group()
                numerator = gl.amd.cdna4.mfma(
                    p,
                    gl.convert_layout(value, rhs_layout),
                    numerators[panel] * alpha_mma[:, None],
                )
                next_numerators += (numerator,)
            numerators = next_numerators
            maximum = next_maximum
        if step == 0:
            if active1:
                if not active0:
                    _prefetch_panel(
                        successor_panels[0],
                        KV,
                        slots1,
                        KV_ROW,
                        0,
                        BLOCK_N,
                        USE_U24=USE_U24,
                    )
                    gl.amd.cdna4.async_copy.commit_group()
                    _prefetch_panel(
                        successor_panels[4],
                        KV,
                        slots1,
                        KV_ROW,
                        512,
                        BLOCK_N,
                        USE_U24=USE_U24,
                    )
                    gl.amd.cdna4.async_copy.commit_group()
                    for panel in gl.static_range(1, 4):
                        _prefetch_panel(
                            successor_panels[panel],
                            KV,
                            slots1,
                            KV_ROW,
                            panel * 128,
                            BLOCK_N,
                            USE_U24=USE_U24,
                        )
                        gl.amd.cdna4.async_copy.commit_group()
    for panel in gl.static_range(4):
        _store_partial_octet(
            numerators[panel],
            Partial,
            row,
            part,
            oh[:, None],
            panel * 128,
            HEADS,
            SPLITS,
            ".wt",
        )
    stats_head = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, softmax_layout))
    stats_base = Stats + ((row * SPLITS + part) * HEADS + stats_head) * 2
    gl.store(stats_base, maximum, stats_head < HEADS)
    gl.store(stats_base + 1, denominator, stats_head < HEADS)


@gluon.jit
def _partial_wide(
    Q,
    KV,
    Slots,
    Partial,
    Stats,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    KV_ROW: gl.constexpr,
    S_ROW: gl.constexpr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    scale,
    BLOCK_N: gl.constexpr,
):
    WIDTHS: gl.constexpr = (256, 256)
    OFFSETS: gl.constexpr = (0, 256)
    PANELS: gl.constexpr = len(WIDTHS)
    row, part, head_group = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    witness0 = gl.load(Slots + row * S_ROW + part * BLOCK_N)
    witness1 = gl.load(Slots + row * S_ROW + (SPLITS + part) * BLOCK_N)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    lhs: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 16)
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma_layout, 8)
    operand: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
    h = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, load_layout))
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
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    n = gl.arange(0, BLOCK_N, gl.SliceLayout(1, copy_layout))
    slots0 = gl.load(Slots + row * S_ROW + part * BLOCK_N + n)
    slots1 = gl.load(Slots + row * S_ROW + (SPLITS + part) * BLOCK_N + n)
    active0, active1 = (witness0 >= 0, witness1 >= 0)
    if not active0:
        active0 = gl.max(slots0, 0) >= 0
    if not active1:
        active1 = gl.max(slots1, 0) >= 0
    key0 = gl.allocate_shared_memory(
        KV.dtype.element_ty,
        [BLOCK_N, WIDTHS[0]],
        gl.SwizzledSharedLayout(16, 1, 16, [1, 0]),
    )
    query_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    query_smem = gl.allocate_shared_memory(gl.bfloat16, [16, 512], query_layout, q)
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], query_layout, qr
    )
    if active0:
        _prefetch_panel(key0, KV, slots0, KV_ROW, 0, BLOCK_N, WIDTHS[0])
    query_panels = ()
    for fragment in gl.static_range(4):
        query_panels += (query_smem.slice(fragment * 128, 128, dim=1).load(lhs),)
    rotary_query = rotary_query_smem.load(lhs)
    query_smem._keep_alive()
    rotary_query_smem._keep_alive()
    key_panels = (key0,)
    for panel in gl.static_range(1, PANELS):
        key_panels += (
            gl.allocate_shared_memory(
                KV.dtype.element_ty,
                [BLOCK_N, WIDTHS[panel]],
                gl.SwizzledSharedLayout(16, 1, 16, [1, 0]),
            ),
        )
    rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    )
    if active0:
        for panel in gl.static_range(1, PANELS):
            _prefetch_panel(
                key_panels[panel],
                KV,
                slots0,
                KV_ROW,
                OFFSETS[panel],
                BLOCK_N,
                WIDTHS[panel],
            )
        _prefetch_panel(rotary, KV, slots0, KV_ROW, 512, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
    numerators = ()
    for fragment in gl.static_range(4):
        numerators += (gl.zeros((16, 128), gl.float32, mma_layout),)
    oh = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma_layout))
    maximum = gl.full(
        (16,), -float("inf"), gl.float32, gl.SliceLayout(1, softmax_layout)
    )
    denominator = gl.zeros((16,), gl.float32, gl.SliceLayout(1, softmax_layout))
    rotary_copy: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    rotary_slots1 = gl.convert_layout(slots1, gl.SliceLayout(1, rotary_copy))
    for step in gl.static_range(2):
        if step == 0:
            slot, active = (slots0, active0)
        else:
            slot, active = (slots1, active1)
        valid = slot >= 0
        if active:
            if step == 1:
                valid_softmax = gl.convert_layout(
                    valid, gl.SliceLayout(0, softmax_layout)
                )
                gl.amd.cdna4.async_copy.wait_group(PANELS)
            else:
                gl.amd.cdna4.async_copy.wait_group(0)
            score = gl.zeros((16, BLOCK_N), gl.float32, mma_layout)
            for panel in gl.static_range(PANELS):
                if step == 1 and panel > 0:
                    gl.amd.cdna4.async_copy.wait_group(PANELS - panel)
                for half in gl.static_range(2):
                    score = _qk_panel(
                        query_panels[2 * panel + half],
                        key_panels[panel].slice(half * 128, 128, dim=1),
                        score,
                        BLOCK_N,
                        step == 1,
                    )
            if step == 1:
                gl.amd.cdna4.async_copy.wait_group(0)
            score = _qk_panel(rotary_query, rotary, score, BLOCK_N, step == 1)
            if step == 0:
                valid_softmax = gl.convert_layout(
                    valid, gl.SliceLayout(0, softmax_layout)
                )
            p, alpha, maximum, denominator = _online_softmax(
                score,
                valid_softmax,
                maximum,
                denominator,
                rotary,
                scale,
                BLOCK_N,
                step == 0,
            )
            next_numerators = ()
            for panel in gl.static_range(PANELS):
                for half in gl.static_range(2):
                    value = (
                        key_panels[panel]
                        .slice(half * 128, 128, dim=1)
                        .load(operand)
                        .to(gl.bfloat16)
                    )
                    if step == 0 and half == 1:
                        if active1:
                            gl.barrier()
                            _prefetch_panel(
                                key_panels[panel],
                                KV,
                                slots1,
                                KV_ROW,
                                OFFSETS[panel],
                                BLOCK_N,
                                WIDTHS[panel],
                            )
                            gl.amd.cdna4.async_copy.commit_group()
                    numerator = gl.amd.cdna4.mfma(
                        p,
                        gl.convert_layout(value, rhs),
                        numerators[2 * panel + half] * alpha[:, None],
                    )
                    next_numerators += (numerator,)
            numerators = next_numerators
        if step == 0:
            if active1:
                if not active0:
                    for panel in gl.static_range(PANELS):
                        _prefetch_panel(
                            key_panels[panel],
                            KV,
                            slots1,
                            KV_ROW,
                            OFFSETS[panel],
                            BLOCK_N,
                            WIDTHS[panel],
                        )
                        gl.amd.cdna4.async_copy.commit_group()
                gl.barrier()
                _prefetch_panel(rotary, KV, rotary_slots1, KV_ROW, 512, BLOCK_N)
                gl.amd.cdna4.async_copy.commit_group()
    for fragment in gl.static_range(4):
        _store_partial_octet(
            numerators[fragment],
            Partial,
            row,
            part,
            oh[:, None],
            fragment * 128,
            HEADS,
            SPLITS,
        )
    sh = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, softmax_layout))
    stats_base = Stats + ((row * SPLITS + part) * HEADS + sh) * 2
    gl.store(stats_base, maximum, sh < HEADS)
    gl.store(stats_base + 1, denominator, sh < HEADS)


@gluon.jit
def _merge_native(
    Partial,
    Stats,
    Out,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    WAVES: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8 if SPLITS == 8 else 4], [GROUP_H, 64 // GROUP_H], [1, WAVES], [0, 1]
    )
    stats_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [GROUP_H, 64 // GROUP_H], [WAVES, 1], [0, 1]
    )
    row = gl.program_id(0)
    h0 = gl.program_id(2) * GROUP_H
    h = h0 + gl.arange(0, GROUP_H, gl.SliceLayout(1, stats_layout))
    split = gl.arange(0, SPLITS, gl.SliceLayout(0, stats_layout))
    maximum = gl.load(
        Stats + ((row * SPLITS + split[None, :]) * HEADS + h[:, None]) * 2
    )
    den = gl.load(
        Stats + ((row * SPLITS + split[None, :]) * HEADS + h[:, None]) * 2 + 1
    )
    maximum_groups = maximum.reshape((GROUP_H, SPLITS // 8, 4, 2))
    maximum_pairs = gl.max(gl.max(maximum_groups, 1), 1)
    global_max = gl.convert_layout(
        gl.max(maximum_pairs, 1), gl.SliceLayout(1, stats_layout), assert_trivial=True
    )
    global_max = gl.where(global_max == -float("inf"), 0.0, global_max)
    weights = gl.exp2((maximum - global_max[:, None]) * 1.4426950408889634)
    denominator_groups = (den * weights).reshape((GROUP_H, SPLITS // 8, 4, 2))
    denominator_pairs = gl.sum(gl.sum(denominator_groups, 1), 1)
    denominator = gl.convert_layout(
        gl.sum(denominator_pairs, 1), gl.SliceLayout(1, layout)
    )
    oh = h0 + gl.arange(0, GROUP_H, gl.SliceLayout(1, layout))
    d = gl.program_id(1) * BLOCK_D + gl.arange(0, BLOCK_D, gl.SliceLayout(0, layout))
    if SPLITS == 8:
        panel = gl.program_id(1) // (128 // BLOCK_D)
        within = gl.program_id(1) % (128 // BLOCK_D) * BLOCK_D
        pd = within + gl.arange(0, BLOCK_D, gl.SliceLayout(0, layout))
        partial_base = Partial + (row * 4 * SPLITS + panel * SPLITS) * HEADS * 128
        partial_offset = (
            pd[None, :] // 8 * HEADS * 8 + oh[:, None] * 8 + pd[None, :] % 8
        )
    else:
        panel = gl.program_id(1) // (128 // BLOCK_D)
        within = gl.program_id(1) % (128 // BLOCK_D) * BLOCK_D
        pd = within + gl.arange(0, BLOCK_D, gl.SliceLayout(0, layout))
        partial_base = Partial + (row * 4 + panel) * SPLITS * HEADS * 128
        partial_offset = (
            pd[None, :] % 64 // 4 * HEADS * 8
            + oh[:, None] * 8
            + pd[None, :] // 64 * 4
            + pd[None, :] % 4
        ).to(gl.uint32)
    acc = gl.zeros((GROUP_H, BLOCK_D), gl.float32, layout)
    for part in gl.static_range(SPLITS):
        index = gl.full((GROUP_H, 1), part, gl.int32, stats_layout)
        weight = gl.convert_layout(
            gl.sum(gl.gather(weights, index, 1), 1), gl.SliceLayout(1, layout)
        )
        p = gl.amd.cdna4.buffer_load(
            partial_base, part * HEADS * 128 + partial_offset, cache=".cg"
        ).to(gl.float32)
        acc = gl.fma(p, weight[:, None], acc)
    reciprocal = 1.0 / gl.where(denominator > 0.0, denominator, 1.0)
    result = acc * reciprocal[:, None]
    if SPLITS == 8:
        d = gl.program_id(1) * BLOCK_D + pd // 8 * 4 + pd % 8 // 4 * 64 + pd % 4
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
    block_n = 64 if m <= 32 else 128
    splits = selected // (2 * block_n)
    record_width = 8
    partial = torch.empty(
        (m, 4, splits, 128 // record_width, heads, record_width),
        device=query.device,
        dtype=torch.bfloat16,
    )
    stats = torch.empty((m, splits, heads, 2), device=query.device, dtype=torch.float32)
    output = torch.empty((m, heads, 512), device=query.device, dtype=torch.bfloat16)
    producer_args = (
        query,
        kv_cache,
        selected_slots,
        partial,
        stats,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        heads,
        splits,
        softmax_scale,
        block_n,
    )
    grid = (m, splits, 1)
    if block_n == 64:
        compact_offsets = (
            kv_cache.shape[0] <= 1 << 24
            and 0 < kv_cache.stride(0) < 1 << 24
            and (kv_cache.stride(0) % 16 == 0)
            and (kv_cache.shape[0] * kv_cache.stride(0) < 1 << 31)
        )
        _partial_attention[grid](
            *producer_args,
            USE_U24=compact_offsets,
            num_warps=4,
            num_stages=1,
            waves_per_eu=0,
        )
    else:
        _partial_wide[grid](*producer_args, num_warps=4, num_stages=1, waves_per_eu=2)
    group_h, block_d, waves = (8, 64, 2) if m <= 32 else (8, 128, 2)
    _merge_native[m, 512 // block_d, heads // group_h](
        partial, stats, output, heads, splits, group_h, block_d, waves, num_warps=waves
    )
    return output
