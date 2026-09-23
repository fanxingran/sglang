# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp8_m64.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _prefetch_panel(
    dest, KV, slots, KV_ROW: gl.constexpr, CHANNEL: gl.constexpr, BLOCK_N: gl.constexpr
):
    if CHANNEL == 512:
        WIDTH: gl.constexpr = 64
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
        slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
    else:
        WIDTH: gl.constexpr = 128
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    columns = gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    rows = gl.arange(0, BLOCK_N, gl.SliceLayout(1, layout))
    flat = dest._reinterpret(
        KV.dtype.element_ty, [BLOCK_N, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    physical_columns = columns[None, :] ^ rows[:, None] % (WIDTH // 16) * 16
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        flat,
        KV,
        slots[:, None] * KV_ROW
        + CHANNEL
        + gl.max_contiguous(gl.multiple_of(physical_columns, [1, 16]), [1, 16]),
        slots[:, None] >= 0,
        cache_modifier=".cv",
    )


@gluon.jit
def _qk_panel(query, key_smem, score, RELAXED: gl.constexpr):
    operand_layout: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, 4], [1, 0])
    rhs_layout: gl.constexpr = gl.DotOperandLayout(1, score.type.layout, 16)
    if RELAXED:
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
    FIRST_TILE: gl.constexpr,
):
    mma_layout: gl.constexpr = score.type.layout
    softmax_layout: gl.constexpr = valid.type.layout.parent
    lhs_layout: gl.constexpr = gl.DotOperandLayout(0, mma_layout, 8)
    query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    score_smem = rotary_smem._reinterpret(
        gl.float32, [16, BLOCK_N], gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    score_smem.store(score)
    score = score_smem.load(softmax_layout)
    score_smem._keep_alive()
    score = gl.where(valid[None, :], score * scale, -float("inf"))
    next_maximum = gl.maximum(maximum, gl.max(score, 1))
    safe_maximum = gl.where(next_maximum == -float("inf"), 0.0, next_maximum)
    if not FIRST_TILE:
        alpha = gl.exp2((maximum - safe_maximum) * 1.4426950408889634)
    probability = gl.where(
        valid[None, :],
        gl.exp2((score - safe_maximum[:, None]) * 1.4426950408889634),
        0.0,
    )
    if FIRST_TILE:
        denominator = gl.sum(probability, 1)
    else:
        denominator = denominator * alpha + gl.sum(probability, 1)
    probability_smem = rotary_smem._reinterpret(
        gl.bfloat16, [32, BLOCK_N], query_shared_layout
    ).slice(0, 16, dim=0)
    if not FIRST_TILE:
        alpha_smem = rotary_smem._reinterpret(
            gl.float32, [16 * BLOCK_N], gl.SwizzledSharedLayout(1, 1, 1, [0])
        ).slice(8 * BLOCK_N, 16, dim=0)
        alpha_smem.store(alpha)
    probability_smem.store(probability.to(gl.bfloat16))
    p = probability_smem.load(lhs_layout)
    if not FIRST_TILE:
        alpha_mma = alpha_smem.load(gl.SliceLayout(1, mma_layout))
        alpha_smem._keep_alive()
    else:
        alpha_mma = gl.full((16,), 0.0, gl.float32, gl.SliceLayout(1, mma_layout))
    probability_smem._keep_alive()
    return (p, alpha_mma, next_maximum, denominator)


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
):
    row = gl.program_id(0)
    part = gl.program_id(1)
    head_group = gl.program_id(2)
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
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
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
    query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    if active0:
        _prefetch_panel(key0_smem, KV, slots0, KV_ROW, 0, BLOCK_N)
        gl.amd.cdna4.async_copy.commit_group()
        _prefetch_panel(key1_smem, KV, slots0, KV_ROW, 128, BLOCK_N)
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
    od = gl.arange(0, 128, gl.SliceLayout(0, mma_layout))
    partial_base = Partial + row * SPLITS * HEADS * 512 + part * 128 * HEADS
    partial_offsets = od[None, :] // 4 * HEADS * 4 + oh[:, None] * 4 + od[None, :] % 4
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
    successor_key0 = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 128], shared_layout
    )
    successor_rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [BLOCK_N, 64], rotary_shared_layout
    )
    successor_panels = (
        successor_key0,
        key1_smem,
        key2_smem,
        key3_smem,
        successor_rotary,
    )
    if active0:
        for panel in gl.static_range(2, 5):
            _prefetch_panel(key_panels[panel], KV, slots0, KV_ROW, panel * 128, BLOCK_N)
            gl.amd.cdna4.async_copy.commit_group()
    for step in gl.static_range(2):
        if step == 0:
            slot = slots0
            any_valid = active0
            current_panels = key_panels
        else:
            slot = slots1
            any_valid = active1
            current_panels = successor_panels
        valid = slot >= 0
        if any_valid:
            gl.amd.cdna4.async_copy.wait_group(4)
            score = gl.zeros((16, BLOCK_N), gl.float32, mma_layout)
            for panel in gl.static_range(5):
                if step == 0 and panel > 0:
                    gl.amd.cdna4.async_copy.wait_group(4 - panel)
                if step == 1 and panel > 0:
                    gl.amd.cdna4.async_copy.wait_group(3 - panel if panel < 4 else 0)
                score = _qk_panel(
                    query_panels[panel], current_panels[panel], score, True
                )
            valid_softmax = gl.convert_layout(valid, gl.SliceLayout(0, softmax_layout))
            if step == 0:
                if active1:
                    _prefetch_panel(successor_key0, KV, slots1, KV_ROW, 0, BLOCK_N)
                    gl.amd.cdna4.async_copy.commit_group()
                    _prefetch_panel(successor_rotary, KV, slots1, KV_ROW, 512, BLOCK_N)
                    gl.amd.cdna4.async_copy.commit_group()
            p, alpha_mma, next_maximum, denominator = _online_softmax(
                score,
                valid_softmax,
                maximum,
                denominator,
                current_panels[4],
                scale,
                BLOCK_N,
                step == 0,
            )
            next_numerators = ()
            for panel in gl.static_range(4):
                value = current_panels[panel].load(operand_blocked).to(gl.bfloat16)
                if step == 0:
                    if active1:
                        if panel > 0:
                            gl.barrier()
                            _prefetch_panel(
                                successor_panels[panel],
                                KV,
                                slots1,
                                KV_ROW,
                                panel * 128,
                                BLOCK_N,
                            )
                            if panel < 3:
                                gl.amd.cdna4.async_copy.commit_group()
                numerator = gl.amd.cdna4.mfma(
                    p,
                    gl.convert_layout(value, rhs_layout),
                    numerators[panel] * alpha_mma[:, None],
                )
                next_numerators += (numerator,)
            if step == 1:
                for panel in gl.static_range(4):
                    gl.amd.cdna4.buffer_store(
                        next_numerators[panel].to(gl.bfloat16),
                        partial_base,
                        partial_offsets + panel * 128 * SPLITS * HEADS,
                        oh[:, None] < HEADS,
                        cache=".wt",
                    )
            if step == 0:
                numerators = next_numerators
            maximum = next_maximum
        if step == 0:
            if active1:
                if not active0:
                    _prefetch_panel(successor_key0, KV, slots1, KV_ROW, 0, BLOCK_N)
                    gl.amd.cdna4.async_copy.commit_group()
                    _prefetch_panel(successor_rotary, KV, slots1, KV_ROW, 512, BLOCK_N)
                    gl.amd.cdna4.async_copy.commit_group()
                    for panel in gl.static_range(1, 4):
                        _prefetch_panel(
                            successor_panels[panel],
                            KV,
                            slots1,
                            KV_ROW,
                            panel * 128,
                            BLOCK_N,
                        )
                        if panel < 3:
                            gl.amd.cdna4.async_copy.commit_group()
                gl.amd.cdna4.async_copy.commit_group()
    if not any_valid:
        for panel in gl.static_range(4):
            gl.amd.cdna4.buffer_store(
                numerators[panel].to(gl.bfloat16),
                partial_base,
                partial_offsets + panel * 128 * SPLITS * HEADS,
                oh[:, None] < HEADS,
                cache=".wt",
            )
    stats_head = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, softmax_layout))
    stats_base = Stats + (row * 2 * SPLITS + part) * HEADS + stats_head
    gl.store(stats_base, maximum, stats_head < HEADS)
    gl.store(stats_base + SPLITS * HEADS, denominator, stats_head < HEADS)


@gluon.jit
def _prefetch_wide(
    dest, KV, slots, KV_ROW: gl.constexpr, CHANNEL: gl.constexpr, COMPACT: gl.constexpr
):
    if CHANNEL == 512:
        WIDTH: gl.constexpr = 64
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
        CACHE: gl.constexpr = ""
    else:
        WIDTH: gl.constexpr = 256
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
        CACHE: gl.constexpr = ".cg"
    slots = gl.convert_layout(slots, gl.SliceLayout(1, layout))
    columns = gl.arange(0, WIDTH, gl.SliceLayout(0, layout))
    rows = gl.arange(0, 128, gl.SliceLayout(1, layout))
    flat = dest._reinterpret(
        KV.dtype.element_ty, [128, WIDTH], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    physical_columns = columns[None, :] ^ rows[:, None] % (WIDTH // 16) * 16
    columns_swizzled = gl.max_contiguous(
        gl.multiple_of(physical_columns, [1, 16]), [1, 16]
    )
    if COMPACT:
        offsets = gl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3;",
            constraints="=v,v,s,v",
            args=[slots[:, None].to(gl.int32), KV_ROW, columns_swizzled + CHANNEL],
            dtype=gl.int32,
            is_pure=True,
            pack=1,
        )
        offsets = gl.max_contiguous(gl.multiple_of(offsets, [1, 16]), [1, 16])
    else:
        offsets = slots[:, None] * KV_ROW + CHANNEL + columns_swizzled
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        flat, KV, offsets, slots[:, None] >= 0, cache_modifier=CACHE
    )


@gluon.jit
def _publish_octets(numerator, Partial, base_offset, head_group, HEADS: gl.constexpr):
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [1, 4], [0, 1])
    packed = numerator.to(gl.bfloat16).reshape((16, 2, 16, 4))
    packed = packed.permute((0, 2, 1, 3)).reshape((16, 128))
    packed = gl.convert_layout(packed, packed_layout, assert_trivial=True)
    h = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, packed_layout))
    c = gl.arange(0, 128, gl.SliceLayout(0, packed_layout))
    offsets = c[None, :] // 8 * HEADS * 8 + h[:, None] * 8 + c[None, :] % 8
    gl.amd.cdna4.buffer_store(
        packed, Partial + base_offset, offsets, h[:, None] < HEADS, cache=".cs"
    )


@gluon.jit
def _partial_attention_wide(
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
    COMPACT: gl.constexpr,
):
    row = gl.program_id(0)
    part = gl.program_id(1)
    head_group = gl.program_id(2)
    witness0 = gl.load(Slots + row * S_ROW + part * 128)
    witness1 = gl.load(Slots + row * S_ROW + (SPLITS + part) * 128)
    load_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    mma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    softmax_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    operand_layout: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [1, 0])
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
    copy_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    cn = gl.arange(0, 128, gl.SliceLayout(1, copy_layout))
    slots0 = gl.amd.cdna4.buffer_load(Slots + row * S_ROW + part * 128, cn)
    slots1 = gl.amd.cdna4.buffer_load(Slots + row * S_ROW + (SPLITS + part) * 128, cn)
    active0 = witness0 >= 0
    active1 = witness1 >= 0
    if not active0:
        active0 = gl.max(slots0, 0) >= 0
    if not active1:
        active1 = gl.max(slots1, 0) >= 0
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    key0 = gl.allocate_shared_memory(KV.dtype.element_ty, [128, 256], shared_layout)
    query_shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 512], query_shared_layout, q
    )
    rotary_query_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 64], query_shared_layout, qr
    )
    if active0:
        _prefetch_wide(key0, KV, slots0, KV_ROW, 0, COMPACT)
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
    maximum = gl.full(
        (16,), -float("inf"), gl.float32, gl.SliceLayout(1, softmax_layout)
    )
    denominator = gl.zeros((16,), gl.float32, gl.SliceLayout(1, softmax_layout))
    key1 = gl.allocate_shared_memory(KV.dtype.element_ty, [128, 256], shared_layout)
    rotary = gl.allocate_shared_memory(
        KV.dtype.element_ty, [128, 64], gl.SwizzledSharedLayout(16, 1, 4, [1, 0])
    )
    key_panels = (key0, key1, rotary)
    if active0:
        _prefetch_wide(key1, KV, slots0, KV_ROW, 256, COMPACT)
        _prefetch_wide(rotary, KV, slots0, KV_ROW, 512, COMPACT)
        gl.amd.cdna4.async_copy.commit_group()
    rotary_copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [16, 4], [4, 1], [1, 0]
    )
    rotary_slots1 = gl.convert_layout(slots1, gl.SliceLayout(1, rotary_copy_layout))
    for step in gl.static_range(2):
        if step == 0:
            slot = slots0
            any_valid = active0
        else:
            slot = slots1
            any_valid = active1
        if any_valid:
            valid = slot >= 0
            if step == 1:
                valid_softmax = gl.convert_layout(
                    valid, gl.SliceLayout(0, softmax_layout)
                )
                gl.amd.cdna4.async_copy.wait_group(2)
            else:
                gl.amd.cdna4.async_copy.wait_group(0)
            score = gl.zeros((16, 128), gl.float32, mma_layout)
            for panel in gl.static_range(5):
                if step == 1 and panel == 2:
                    gl.amd.cdna4.async_copy.wait_group(1)
                if step == 1 and panel == 4:
                    gl.amd.cdna4.async_copy.wait_group(0)
                if panel < 4:
                    key_view = key_panels[panel // 2].slice(panel % 2 * 128, 128, dim=1)
                else:
                    key_view = rotary
                score = _qk_panel(query_panels[panel], key_view, score, step == 1)
            if step == 0:
                valid_softmax = gl.convert_layout(
                    valid, gl.SliceLayout(0, softmax_layout)
                )
            p, alpha, next_maximum, denominator = _online_softmax(
                score,
                valid_softmax,
                maximum,
                denominator,
                rotary,
                scale,
                128,
                step == 0,
            )
            next_numerators = ()
            for panel in gl.static_range(4):
                value = (
                    key_panels[panel // 2]
                    .slice(panel % 2 * 128, 128, dim=1)
                    .load(operand_layout)
                    .to(gl.bfloat16)
                )
                if step == 0:
                    if active1 and panel % 2 == 1:
                        gl.barrier()
                        _prefetch_wide(
                            key_panels[panel // 2],
                            KV,
                            slots1,
                            KV_ROW,
                            panel // 2 * 256,
                            COMPACT,
                        )
                        gl.amd.cdna4.async_copy.commit_group()
                        if panel == 3:
                            _prefetch_wide(
                                rotary, KV, rotary_slots1, KV_ROW, 512, COMPACT
                            )
                            gl.amd.cdna4.async_copy.commit_group()
                numerator = gl.amd.cdna4.mfma(
                    p,
                    gl.convert_layout(value, rhs_layout),
                    numerators[panel] * alpha[:, None],
                )
                if step == 1:
                    _publish_octets(
                        numerator,
                        Partial,
                        (row * SPLITS + part) * HEADS * 512 + panel * 128 * HEADS,
                        head_group,
                        HEADS,
                    )
                if step == 0:
                    next_numerators += (numerator,)
            if step == 0:
                numerators = next_numerators
            maximum = next_maximum
        if step == 0:
            if active1 and (not active0):
                for panel in gl.static_range(2):
                    _prefetch_wide(
                        key_panels[panel], KV, slots1, KV_ROW, panel * 256, COMPACT
                    )
                    gl.amd.cdna4.async_copy.commit_group()
                    if panel == 1:
                        _prefetch_wide(rotary, KV, rotary_slots1, KV_ROW, 512, COMPACT)
                        gl.amd.cdna4.async_copy.commit_group()
    if not any_valid:
        for panel in gl.static_range(4):
            _publish_octets(
                numerators[panel],
                Partial,
                (row * SPLITS + part) * HEADS * 512 + panel * 128 * HEADS,
                head_group,
                HEADS,
            )
    stats_head = head_group * 16 + gl.arange(0, 16, gl.SliceLayout(1, softmax_layout))
    stats_base = Stats + (row * 2 * SPLITS + part) * HEADS + stats_head
    gl.store(stats_base, maximum, stats_head < HEADS)
    gl.store(stats_base + SPLITS * HEADS, denominator, stats_head < HEADS)


@gluon.jit
def _merge_native(
    Partial,
    Stats,
    Out,
    SPLITS: gl.constexpr,
    HEADS: gl.constexpr,
    RECORD_GROUP: gl.constexpr,
):
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [2, 1], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 2], [0, 1])
    row = gl.program_id(0)
    h = gl.arange(0, 8, gl.SliceLayout(1, stats_layout))
    s = gl.arange(0, SPLITS, gl.SliceLayout(0, stats_layout))
    stats_base = Stats + row * 2 * SPLITS * HEADS + s[None, :] * HEADS + h[:, None]
    maxima = gl.load(stats_base)
    denominators = gl.load(stats_base + SPLITS * HEADS)
    native_maxima = maxima.reshape((8, 2, 4, 2))
    maximum = gl.max(gl.max(gl.max(native_maxima, 1), 1), 1)
    maximum = gl.convert_layout(
        maximum, gl.SliceLayout(1, stats_layout), assert_trivial=True
    )
    maximum = gl.where(maximum == -float("inf"), 0.0, maximum)
    weights = gl.exp2((maxima - maximum[:, None]) * 1.4426950408889634)
    terms = denominators * weights
    denominator = gl.sum(gl.sum(gl.sum(terms.reshape((8, 2, 4, 2)), 1), 1), 1)
    denominator = gl.convert_layout(denominator, gl.SliceLayout(1, output_layout))
    oh = gl.convert_layout(h, gl.SliceLayout(1, output_layout))
    d = gl.program_id(1) * 64 + gl.arange(0, 64, gl.SliceLayout(0, output_layout))
    acc = gl.zeros((8, 64), gl.float32, output_layout)
    offsets = (
        d[None, :] // RECORD_GROUP * SPLITS * RECORD_GROUP * HEADS
        + d[None, :] % RECORD_GROUP // 4 * HEADS * 4
        + oh[:, None] * 4
        + d[None, :] % 4
    )
    for split in gl.static_range(SPLITS):
        index = gl.full((8, 1), split, gl.int32, stats_layout)
        weight = gl.sum(gl.gather(weights, index, 1), 1)
        weight = gl.convert_layout(weight, gl.SliceLayout(1, output_layout))
        partial_base = (
            Partial + row * SPLITS * HEADS * 512 + split * RECORD_GROUP * HEADS
        )
        partial = gl.load(partial_base + offsets, cache_modifier=".cg").to(gl.float32)
        acc = gl.fma(partial, weight[:, None], acc)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    result = acc * inverse_denominator[:, None]
    gl.store(
        Out + (row * HEADS + oh[:, None]) * 512 + d[None, :], result.to(gl.bfloat16)
    )


@gluon.jit
def _merge_octets(Partial, Stats, Out, SPLITS: gl.constexpr, HEADS: gl.constexpr):
    stats_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [2, 1], [0, 1])
    packed_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [0, 1])
    row = gl.program_id(0)
    panel = gl.program_id(1)
    h = gl.arange(0, 8, gl.SliceLayout(1, stats_layout))
    s = gl.arange(0, SPLITS, gl.SliceLayout(0, stats_layout))
    stats_base = Stats + row * 2 * SPLITS * HEADS + s[None, :] * HEADS + h[:, None]
    maxima = gl.load(stats_base)
    denominators = gl.load(stats_base + SPLITS * HEADS)
    native_maxima = maxima.reshape((8, 4, 2))
    maximum = gl.max(gl.max(native_maxima, 1), 1)
    maximum = gl.convert_layout(
        maximum, gl.SliceLayout(1, stats_layout), assert_trivial=True
    )
    maximum = gl.where(maximum == -float("inf"), 0.0, maximum)
    weights = gl.exp2((maxima - maximum[:, None]) * 1.4426950408889634)
    terms = denominators * weights
    denominator = gl.sum(gl.sum(terms.reshape((8, 4, 2)), 1), 1)
    denominator = gl.convert_layout(denominator, gl.SliceLayout(1, packed_layout))
    oh = gl.convert_layout(h, gl.SliceLayout(1, packed_layout))
    c = gl.arange(0, 128, gl.SliceLayout(0, packed_layout))
    offsets = c[None, :] // 8 * HEADS * 8 + oh[:, None] * 8 + c[None, :] % 8
    acc = gl.zeros((8, 128), gl.float32, packed_layout)
    for split in gl.static_range(SPLITS):
        index = gl.full((8, 1), split, gl.int32, stats_layout)
        weight = gl.sum(gl.gather(weights, index, 1), 1)
        weight = gl.convert_layout(weight, gl.SliceLayout(1, packed_layout))
        partial_base = (
            Partial + (row * SPLITS + split) * HEADS * 512 + panel * HEADS * 128
        )
        partial = gl.amd.cdna4.buffer_load(partial_base, offsets, cache=".cg").to(
            gl.float32
        )
        acc = gl.fma(partial, weight[:, None], acc)
    inverse_denominator = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    result = acc * inverse_denominator[:, None]
    d = panel * 128 + c // 8 * 4 + c % 8 // 4 * 64 + c % 4
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
    block_n = 64 if m <= 40 else 128
    splits = selected // (2 * block_n)
    if block_n == 64:
        partial_shape = (m, 4, splits, 32, heads, 4)
    else:
        partial_shape = (m, splits, 4, 16, heads, 8)
    partial = torch.empty(partial_shape, device=query.device, dtype=torch.bfloat16)
    stats = torch.empty((m, 2, splits, heads), device=query.device, dtype=torch.float32)
    output = torch.empty((m, heads, 512), device=query.device, dtype=torch.bfloat16)
    if block_n == 64:
        _partial_attention[m, splits, triton.cdiv(heads, 16)](
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
            num_warps=4,
            num_stages=1,
        )
    else:
        compact_offsets = (
            kv_cache.shape[0] < 1 << 24
            and 0 < kv_cache.stride(0) < 1 << 24
            and (kv_cache.shape[0] * kv_cache.stride(0) < 1 << 31)
        )
        _partial_attention_wide[m, splits, triton.cdiv(heads, 16)](
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
            compact_offsets,
            num_warps=4,
            num_stages=1,
            waves_per_eu=2,
        )
    if block_n == 64:
        _merge_native[m, 8](partial, stats, output, splits, heads, 128, num_warps=2)
    else:
        _merge_octets[m, 4](partial, stats, output, splits, heads, num_warps=2)
    return output
