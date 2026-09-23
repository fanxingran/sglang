# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m32.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _bitwise_or(a, b):
    return a | b


@gluon.jit
def _wave_ballot(predicate):
    ballot = gl.inline_asm_elementwise(
        "v_cmp_ne_u32 $0, 0, $1",
        constraints="=s,v",
        args=[predicate.to(gl.int32)],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    return ballot


@gluon.jit
def _wave_vote(predicate):
    return (_wave_ballot(predicate) != 0).to(gl.int32)


@gluon.jit
def _replicated_any(predicate, WARPS: gl.constexpr):
    votes = _wave_vote(predicate)
    leader = gl.gather(
        votes, gl.full((1,), 0, gl.int32, gl.BlockedLayout([1], [64], [WARPS], [0])), 0
    )
    return gl.sum(leader, 0) != 0


@gluon.jit
def _two_tile_status(first_active, second_active):
    first = _wave_ballot(first_active).to(gl.uint16)
    second = _wave_ballot(second_active).to(gl.uint16)
    status = (
        (first != 0).to(gl.int32)
        | (second != 0).to(gl.int32) << 1
        | (first != 65535).to(gl.int32) << 2
        | (second != 65535).to(gl.int32) << 3
    )
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    words = gl.convert_layout(status.reshape((4, 16)), layout, assert_trivial=True)
    leaders = gl.gather(words, gl.full((4, 1), 0, gl.int32, layout), 1).reshape((4,))
    return gl.reduce(leaders, 0, _bitwise_or)


@gluon.jit
def _score_mask(active, MMA: gl.constexpr, LOCAL_CORRECTIONS: gl.constexpr):
    if not LOCAL_CORRECTIONS:
        return gl.convert_layout(active, gl.SliceLayout(0, MMA))
    bit_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    ballots = _wave_ballot(active).to(gl.uint32)
    words = gl.convert_layout(ballots.reshape((4, 16)), bit_layout, assert_trivial=True)
    word = gl.gather(words, gl.full((4, 1), 0, gl.int32, bit_layout), 1).reshape((4,))
    ownership = (
        gl.full((16, 64), 0, gl.int32, MMA).reshape((16, 4, 16)).permute((1, 0, 2))
    )
    mask_layout: gl.constexpr = gl.SliceLayout(1, ownership.type.layout)
    word = gl.convert_layout(word, gl.SliceLayout(1, mask_layout), assert_trivial=True)
    token = gl.arange(0, 16, gl.SliceLayout(0, mask_layout))
    mask = word[:, None] >> token[None, :] & 1 != 0
    return gl.convert_layout(
        mask.reshape((64,)), gl.SliceLayout(0, MMA), assert_trivial=True
    )


@gluon.jit
def _merge_any(flags, REDUCE: gl.constexpr):
    per_lane = gl.max((flags & 2).reshape((REDUCE // 2, 32)), 0)
    return _replicated_any(per_lane, 1)


@gluon.jit
def _load_slots(
    SLOTS,
    row,
    position,
    SLOTS_ROW: gl.constexpr,
    SELECTED: gl.constexpr,
    FULL: gl.constexpr = False,
):
    if FULL:
        return gl.load(SLOTS + row * SLOTS_ROW + position).to(gl.int64)
    return gl.load(SLOTS + row * SLOTS_ROW + position, position < SELECTED, -1).to(
        gl.int64
    )


@gluon.jit
def _compensated_pv(
    probability,
    value_memory,
    accumulator,
    LOCAL_CORRECTIONS: gl.constexpr,
    TERMINAL: gl.constexpr,
):
    mma: gl.constexpr = accumulator.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    high = probability.to(gl.bfloat16)
    low = (probability - high.to(gl.float32)).to(gl.bfloat16)
    probability_pair = gl.join(high, low).permute((0, 2, 1)).reshape((16, 128))
    if LOCAL_CORRECTIONS:
        probability_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[128, 8]], [16, 128], [1, 0]
        )
    else:
        probability_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
    if LOCAL_CORRECTIONS:
        value = value_memory.load(dot_b).to(gl.bfloat16)
    probability_memory = gl.allocate_shared_memory(
        gl.bfloat16, (16, 128), probability_shared, probability_pair
    )
    if not LOCAL_CORRECTIONS:
        value = value_memory.load(dot_b).to(gl.bfloat16)
    if not LOCAL_CORRECTIONS and (not TERMINAL):
        value = gl.inline_asm_elementwise(
            "",
            constraints="=v,=v,=v,=v,0,1,2,3",
            args=[value],
            dtype=gl.bfloat16,
            is_pure=True,
            pack=8,
        )
    else:
        value = gl.inline_asm_elementwise(
            "",
            constraints="=v,=v,0,1",
            args=[value],
            dtype=gl.bfloat16,
            is_pure=True,
            pack=4,
        )
    value_pair = gl.join(value, value).permute((2, 0, 1)).reshape((128, 512))
    probability_pair = probability_memory.load(dot_a)
    value_pair = gl.convert_layout(value_pair, dot_b)
    return gl.amd.cdna4.mfma(probability_pair, value_pair, accumulator)


@gluon.jit
def _gather_cache(
    KV,
    slot,
    active,
    latent,
    rotary,
    KV_ROW: gl.constexpr,
    BUFFER_LATENT: gl.constexpr = False,
):
    if BUFFER_LATENT:
        kv = gl.amd.cdna4.buffer_load(
            KV, slot[:, None] * KV_ROW + latent[None, :], active[:, None], 0.0
        )
    else:
        kv = gl.load(
            KV + slot[:, None] * KV_ROW + latent[None, :], active[:, None], 0.0
        )
    kr = gl.amd.cdna4.buffer_load(
        KV + 512, slot[:, None] * KV_ROW + rotary[None, :], active[:, None], 0.0
    )
    return (kv, kr)


@gluon.jit
def _gather_dense(KV, slot, latent, rotary, KV_ROW: gl.constexpr):
    kv = gl.load(KV + slot[:, None] * KV_ROW + latent[None, :])
    kr = gl.load(KV + slot[:, None] * KV_ROW + 512 + rotary[None, :])
    return (kv, kr)


@gluon.jit
def _attention_tile(
    q,
    qr,
    kv,
    kr,
    score_mask,
    scale,
    maximum,
    denominator,
    accumulator,
    tile,
    LOCAL_CORRECTIONS: gl.constexpr,
    latent_shared: gl.constexpr,
    STAGED_LATENT_KEY: gl.constexpr = False,
):
    mma: gl.constexpr = accumulator.type.layout
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    kmem = gl.allocate_shared_memory(kv.dtype, (64, 512), latent_shared, kv)
    if STAGED_LATENT_KEY:
        kt = kmem.permute((1, 0)).load(dot_b).to(gl.bfloat16)
    else:
        kt = gl.convert_layout(kv.permute((1, 0)), dot_b, assert_trivial=True).to(
            gl.bfloat16
        )
    krt = gl.convert_layout(kr.permute((1, 0)), dot_b, assert_trivial=True).to(
        gl.bfloat16
    )
    score = gl.amd.cdna4.mfma(q, kt, gl.full((16, 64), 0.0, gl.float32, mma))
    score = gl.amd.cdna4.mfma(qr, krt, score) * scale
    score = gl.where(score_mask[None, :], score, -float("inf"))
    next_max = gl.maximum(maximum, gl.max(score, 1))
    alpha = gl.exp2((maximum - next_max) * 1.4426950408889634)
    probability = gl.where(
        score_mask[None, :],
        gl.exp2((score - next_max[:, None]) * 1.4426950408889634),
        0.0,
    )
    wave_probability = probability.reshape((16, 4, 16)).permute((1, 0, 2))
    tile_denominator = gl.convert_layout(
        gl.sum(wave_probability, 2), denominator.type.layout, assert_trivial=True
    )
    wave_alpha = gl.convert_layout(
        alpha, gl.SliceLayout(0, denominator.type.layout), assert_trivial=True
    )
    if tile == 0:
        denominator = tile_denominator
        accumulator = gl.full((16, 512), 0.0, gl.float32, mma)
    else:
        denominator = denominator * wave_alpha[None, :] + tile_denominator
        accumulator = accumulator * alpha[:, None]
    accumulator = _compensated_pv(
        probability, kmem, accumulator, LOCAL_CORRECTIONS, STAGED_LATENT_KEY
    )
    return (next_max, denominator, accumulator)


@gluon.jit
def _record_offset(
    row,
    split,
    head,
    channel,
    SPLITS: gl.constexpr,
    RECORD_GROUP: gl.constexpr,
    RECORD_PADDING: gl.constexpr,
):
    return (
        (row * (512 // RECORD_GROUP) + channel // RECORD_GROUP)
        * (SPLITS * 16 * RECORD_GROUP + RECORD_PADDING)
        + split * 16 * RECORD_GROUP
        + head * RECORD_GROUP
        + channel % RECORD_GROUP
    )


@gluon.jit
def _store_partials(
    PARTIAL,
    STATS,
    RESIDUAL,
    row,
    split,
    maximum,
    denominator,
    accumulator,
    SPLITS: gl.constexpr,
    RECORD_GROUP: gl.constexpr,
    LOCAL_CORRECTIONS: gl.constexpr,
    RECORD_PADDING: gl.constexpr,
    LEADER_ROW_PADDING: gl.constexpr,
):
    mma: gl.constexpr = accumulator.type.layout
    heads_out = gl.arange(0, 16, gl.SliceLayout(1, mma))
    values_out = gl.arange(0, 512, gl.SliceLayout(0, mma))
    record = (row * SPLITS + split) * 16 + heads_out
    store_offset = _record_offset(
        row,
        split,
        heads_out[:, None],
        values_out[None, :],
        SPLITS,
        RECORD_GROUP,
        RECORD_PADDING,
    )
    leading = accumulator.to(gl.float16)
    gl.amd.cdna4.buffer_store(
        leading, PARTIAL, store_offset + row * LEADER_ROW_PADDING, cache=".wt"
    )
    if LOCAL_CORRECTIONS:
        denominator = gl.convert_layout(
            gl.sum(denominator, 0), gl.SliceLayout(1, mma), assert_trivial=True
        )
        wave_accumulator = (
            accumulator.reshape((16, 8, 4, 16))
            .permute((2, 0, 1, 3))
            .reshape((4, 16, 128))
        )
        magnitude = gl.max(gl.abs(wave_accumulator), 2)
        local_denominator = gl.convert_layout(
            denominator, gl.SliceLayout(0, magnitude.type.layout), assert_trivial=True
        )
        needs_residual = magnitude > 0.75 * local_denominator[None, :]
        needs_full = needs_residual[:, :, None] & gl.full(
            (4, 16, 128), True, gl.int1, wave_accumulator.type.layout
        )
        needs_full = (
            needs_full.reshape((4, 16, 8, 16)).permute((1, 2, 0, 3)).reshape((16, 512))
        )
        needs_full = gl.convert_layout(needs_full, mma, assert_trivial=True)
        residual = accumulator - leading.to(gl.float32)
        gl.store(RESIDUAL + store_offset, residual, needs_full)
        record = (row * SPLITS + split) * 32 + heads_out
        gl.store(STATS + record, maximum)
        gl.store(STATS + record + 16, denominator)
        ballot = _wave_ballot(needs_residual)
        bit_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        words = gl.convert_layout(ballot.to(gl.uint16), bit_layout, assert_trivial=True)
        packed_flags = gl.gather(
            words, gl.full((4, 1), 0, gl.int32, bit_layout), 1
        ).reshape((4,))
        waves = gl.arange(0, 4, packed_flags.type.layout)
        flag_offset = (
            gl.num_programs(0) * SPLITS * 64 + (row * SPLITS + split) * 4 + waves
        )
        gl.store(STATS.to(gl.pointer_type(gl.uint16)) + flag_offset, packed_flags)
    else:
        wave_accumulator = (
            accumulator.reshape((16, 8, 4, 16))
            .permute((2, 0, 1, 3))
            .reshape((4, 16, 128))
        )
        magnitude = gl.max(gl.abs(wave_accumulator), 2)
        local_denominator = gl.convert_layout(
            denominator, gl.SliceLayout(0, magnitude.type.layout), assert_trivial=True
        )
        local_overflow = magnitude > 65504.0
        local_residual = (magnitude > 0.5 * local_denominator[None, :]) | local_overflow
        correction_bits = _wave_ballot(local_residual).to(gl.uint32) & 65535
        overflow_bits = _wave_ballot(local_overflow).to(gl.uint32) & 65535
        bit_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        words = gl.convert_layout(
            correction_bits | overflow_bits << 16, bit_layout, assert_trivial=True
        )
        wave_words = gl.gather(
            words, gl.full((4, 1), 0, gl.int32, bit_layout), 1
        ).reshape((4,))
        head_bits = gl.reduce(wave_words, 0, _bitwise_or)
        needs_residual = head_bits >> heads_out & 1 != 0
        overflow = head_bits >> heads_out + 16 & 1 != 0
        flags = needs_residual.to(gl.int32) << 1 | overflow.to(gl.int32) << 2
        if head_bits & 65535 != 0:
            residual = accumulator - leading.to(gl.float32)
            residual = gl.where(overflow[:, None], accumulator, residual)
            gl.amd.cdna4.buffer_store(
                residual, RESIDUAL, store_offset, needs_residual[:, None]
            )
        gl.store(STATS + record * 3, maximum)
        gl.store(STATS + record * 3 + 1, denominator)
        gl.store((STATS + record * 3 + 2).to(gl.pointer_type(gl.int32)), flags)


@gluon.jit
def _attention_partials(
    Q,
    KV,
    SLOTS,
    PARTIAL,
    STATS,
    scale,
    Q_ROW: gl.constexpr,
    Q_HEAD: gl.constexpr,
    KV_ROW: gl.constexpr,
    SLOTS_ROW: gl.constexpr,
    SELECTED: gl.constexpr,
    SPLITS: gl.constexpr,
    TILES_PER_SPLIT: gl.constexpr,
    RESIDUAL,
    RECORD_GROUP: gl.constexpr,
    LOCAL_CORRECTIONS: gl.constexpr,
    ROW_GROUP: gl.constexpr,
    RECORD_PADDING: gl.constexpr,
    LEADER_ROW_PADDING: gl.constexpr,
    ROWS: gl.constexpr = 0,
):
    if ROW_GROUP:
        program = gl.program_id(0).to(gl.uint32)
        row_group = program // (ROW_GROUP * SPLITS)
        within_group = program % (ROW_GROUP * SPLITS)
        row = row_group * ROW_GROUP + within_group % ROW_GROUP
        split = within_group // ROW_GROUP
        if ROWS % ROW_GROUP != 0:
            full_programs: gl.constexpr = ROWS // ROW_GROUP * ROW_GROUP * SPLITS
            tail_rows: gl.constexpr = ROWS % ROW_GROUP
            if program >= full_programs:
                tail_program = program - full_programs
                row = ROWS // ROW_GROUP * ROW_GROUP + tail_program % tail_rows
                split = tail_program // tail_rows
    else:
        row = gl.program_id(0)
        split = gl.program_id(1)
    row = row.to(gl.uint32)
    split = split.to(gl.uint32)
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    cache_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    if LOCAL_CORRECTIONS:
        latent_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[512, 16]], [64, 512], [1, 0]
        )
    else:
        latent_shared: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    tokens = gl.arange(0, 64, gl.SliceLayout(1, cache_layout))
    position = split * 64 + tokens
    full_tiles: gl.constexpr = (
        LOCAL_CORRECTIONS and SELECTED == SPLITS * TILES_PER_SPLIT * 64
    )
    next_slot = _load_slots(SLOTS, row, position, SLOTS_ROW, SELECTED, full_tiles)
    if not LOCAL_CORRECTIONS:
        terminal_position = ((TILES_PER_SPLIT - 1) * SPLITS + split) * 64 + tokens
        terminal_slot = _load_slots(SLOTS, row, terminal_position, SLOTS_ROW, SELECTED)
    second_position = (SPLITS + split) * 64 + tokens
    second_slot = _load_slots(
        SLOTS, row, second_position, SLOTS_ROW, SELECTED, full_tiles
    )
    if not LOCAL_CORRECTIONS:
        third_position = (2 * SPLITS + split) * 64 + tokens
        third_slot = _load_slots(SLOTS, row, third_position, SLOTS_ROW, SELECTED)
    heads = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    q_latent = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    q = gl.load(Q + row * Q_ROW + heads[:, None] * Q_HEAD + q_latent[None, :])
    rotary_query_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [16, 4], [4, 1], [0, 1]
    )
    rotary_heads = gl.arange(0, 16, gl.SliceLayout(1, rotary_query_layout))
    q_rotary = gl.arange(0, 64, gl.SliceLayout(0, rotary_query_layout))
    qr = gl.load(
        Q + row * Q_ROW + rotary_heads[:, None] * Q_HEAD + 512 + q_rotary[None, :]
    )
    if not LOCAL_CORRECTIONS:
        terminal_active = (terminal_position < SELECTED) & (terminal_slot >= 0)
        first_active = (position < SELECTED) & (next_slot >= 0)
        second_active = (second_position < SELECTED) & (second_slot >= 0)
        third_active = (third_position < SELECTED) & (third_slot >= 0)
        activity = (
            first_active.to(gl.int32)
            | second_active.to(gl.int32) << 1
            | third_active.to(gl.int32) << 2
            | terminal_active.to(gl.int32) << 3
        )
        activity_bits = gl.reduce(activity, 0, _bitwise_or)
        terminal_any = activity_bits & 8 != 0
    if LOCAL_CORRECTIONS:
        if full_tiles:
            first_active = next_slot >= 0
            second_active = second_slot >= 0
        else:
            first_active = (position < SELECTED) & (next_slot >= 0)
            second_active = (second_position < SELECTED) & (second_slot >= 0)
        activity_bits = _two_tile_status(first_active, second_active)
    qmem = gl.allocate_shared_memory(
        gl.bfloat16, (16, 512), gl.SwizzledSharedLayout(8, 1, 16, [1, 0]), q
    )
    q = qmem.load(dot_a)
    qr = gl.convert_layout(qr, dot_a, assert_trivial=True)
    qmem._keep_alive()
    latent = gl.arange(0, 512, gl.SliceLayout(0, cache_layout))
    rotary = gl.arange(0, 64, gl.SliceLayout(0, cache_layout))
    maximum = gl.full((16,), -float("inf"), gl.float32, gl.SliceLayout(1, mma))
    denominator = gl.full(
        (4, 16), 0.0, gl.float32, gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
    )
    accumulator = gl.full((16, 512), 0.0, gl.float32, mma)
    if not LOCAL_CORRECTIONS:
        slot = next_slot.to(gl.uint32)
        kv, kr = _gather_cache(KV, slot, first_active, latent, rotary, KV_ROW)
        if activity_bits & 1 != 0:
            score_mask = _score_mask(first_active, mma, LOCAL_CORRECTIONS)
            maximum, denominator, accumulator = _attention_tile(
                q,
                qr,
                kv,
                kr,
                score_mask,
                scale,
                maximum,
                denominator,
                accumulator,
                0,
                False,
                latent_shared,
            )
        next_slot = second_slot
    if LOCAL_CORRECTIONS:
        if activity_bits == 3:
            for tile in gl.static_range(0, 2):
                slot = (next_slot if tile == 0 else second_slot).to(gl.uint32)
                kv, kr = _gather_dense(KV, slot, latent, rotary, KV_ROW)
                score_mask = gl.full((64,), True, gl.int1, gl.SliceLayout(0, mma))
                maximum, denominator, accumulator = _attention_tile(
                    q,
                    qr,
                    kv,
                    kr,
                    score_mask,
                    scale,
                    maximum,
                    denominator,
                    accumulator,
                    tile,
                    True,
                    latent_shared,
                )
        else:
            for tile in gl.static_range(0, 2):
                position = (tile * SPLITS + split) * 64 + tokens
                slot = next_slot
                next_slot = second_slot
                if full_tiles:
                    active = slot >= 0
                else:
                    active = (position < SELECTED) & (slot >= 0)
                slot = slot.to(gl.uint32)
                if activity_bits & 1 << tile + 2 == 0:
                    kv, kr = _gather_dense(KV, slot, latent, rotary, KV_ROW)
                    score_mask = gl.full((64,), True, gl.int1, gl.SliceLayout(0, mma))
                    maximum, denominator, accumulator = _attention_tile(
                        q,
                        qr,
                        kv,
                        kr,
                        score_mask,
                        scale,
                        maximum,
                        denominator,
                        accumulator,
                        tile,
                        True,
                        latent_shared,
                    )
                elif activity_bits & 1 << tile != 0:
                    kv, kr = _gather_cache(
                        KV, slot, active, latent, rotary, KV_ROW, True
                    )
                    score_mask = _score_mask(active, mma, True)
                    maximum, denominator, accumulator = _attention_tile(
                        q,
                        qr,
                        kv,
                        kr,
                        score_mask,
                        scale,
                        maximum,
                        denominator,
                        accumulator,
                        tile,
                        True,
                        latent_shared,
                    )
    else:
        for tile in range(1, TILES_PER_SPLIT - 1):
            position = (tile * SPLITS + split) * 64 + tokens
            slot = next_slot
            next_slot = third_slot
            active = (position < SELECTED) & (slot >= 0)
            slot = slot.to(gl.uint32)
            if activity_bits & 1 << tile != 0:
                kv, kr = _gather_cache(KV, slot, active, latent, rotary, KV_ROW)
                score_mask = _score_mask(active, mma, False)
                maximum, denominator, accumulator = _attention_tile(
                    q,
                    qr,
                    kv,
                    kr,
                    score_mask,
                    scale,
                    maximum,
                    denominator,
                    accumulator,
                    tile,
                    False,
                    latent_shared,
                )
    if not LOCAL_CORRECTIONS:
        slot = terminal_slot.to(gl.uint32)
        if terminal_any:
            terminal_layout: gl.constexpr = gl.BlockedLayout(
                [1, 16], [16, 4], [4, 1], [1, 0]
            )
            terminal_latent = gl.arange(0, 512, gl.SliceLayout(0, terminal_layout))
            terminal_index = gl.convert_layout(slot, gl.SliceLayout(1, terminal_layout))
            terminal_valid = gl.convert_layout(
                terminal_active, gl.SliceLayout(1, terminal_layout)
            )
            terminal_kv = gl.load(
                KV + terminal_index[:, None] * KV_ROW + terminal_latent[None, :],
                terminal_valid[:, None],
                0.0,
            )
            terminal_kr = gl.amd.cdna4.buffer_load(
                KV + 512,
                slot[:, None] * KV_ROW + rotary[None, :],
                terminal_active[:, None],
                0.0,
            )
            terminal_mask = _score_mask(terminal_active, mma, False)
            maximum, denominator, accumulator = _attention_tile(
                q,
                qr,
                terminal_kv,
                terminal_kr,
                terminal_mask,
                scale,
                maximum,
                denominator,
                accumulator,
                TILES_PER_SPLIT - 1,
                LOCAL_CORRECTIONS,
                latent_shared,
                True,
            )
    if not LOCAL_CORRECTIONS:
        denominator = gl.convert_layout(
            gl.sum(denominator, 0), gl.SliceLayout(1, mma), assert_trivial=True
        )
    _store_partials(
        PARTIAL,
        STATS,
        RESIDUAL,
        row,
        split,
        maximum,
        denominator,
        accumulator,
        SPLITS,
        RECORD_GROUP,
        LOCAL_CORRECTIONS,
        RECORD_PADDING,
        LEADER_ROW_PADDING,
    )


@gluon.jit
def _merge_native(
    PARTIAL,
    STATS,
    OUTPUT,
    RESIDUAL,
    SPLITS: gl.constexpr,
    REDUCE: gl.constexpr,
    RECORD_GROUP: gl.constexpr,
    LOCAL_CORRECTIONS: gl.constexpr,
    PANEL: gl.constexpr,
    RECORD_PADDING: gl.constexpr,
    LEADER_ROW_PADDING: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [2, 16, 2], [1, 1, 1], [2, 1, 0])
    stats_layout: gl.constexpr = gl.SliceLayout(2, layout)
    splits = gl.arange(0, REDUCE, gl.SliceLayout(1, stats_layout))
    heads = gl.arange(0, 16, gl.SliceLayout(0, stats_layout))
    value_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    values = gl.program_id(1) * PANEL + gl.arange(0, PANEL, value_layout)
    valid = gl.full((REDUCE, 16), True, gl.int1, stats_layout) & (
        splits[:, None] < SPLITS
    )
    if LOCAL_CORRECTIONS:
        record = (row * SPLITS + splits[:, None]) * 32 + heads[None, :]
        maxima = gl.load(STATS + record, valid, -float("inf"))
        denominators = gl.load(STATS + record + 16, valid, 0.0)
        flag_offset = (
            gl.num_programs(0) * SPLITS * 64
            + (row * SPLITS + splits[:, None]) * 4
            + gl.program_id(1) * PANEL // 16 % 4
        )
        flag_words = gl.load(
            STATS.to(gl.pointer_type(gl.uint16)) + flag_offset, valid, 0
        ).to(gl.int32)
        flags = (flag_words >> heads[None, :] & 1) << 1
    else:
        record = (row * SPLITS + splits[:, None]) * 16 + heads[None, :]
        maxima = gl.load(STATS + record * 3, valid, -float("inf"))
        denominators = gl.load(STATS + record * 3 + 1, valid, 0.0)
        flags = gl.load(
            (STATS + record * 3 + 2).to(gl.pointer_type(gl.int32)), valid, 0
        )
    maximum = gl.max(maxima, 0)
    maximum = gl.where(maximum == -float("inf"), 0.0, maximum)
    correction = gl.exp2((maxima - maximum[None, :]) * 1.4426950408889634)
    denominator = gl.sum(denominators * correction, 0)
    split_index = gl.expand_dims(gl.expand_dims(splits, 1), 2)
    head_index = gl.expand_dims(gl.expand_dims(heads, 0), 2)
    value_index = gl.expand_dims(gl.expand_dims(values, 0), 0)
    offsets = _record_offset(
        row, split_index, head_index, value_index, SPLITS, RECORD_GROUP, RECORD_PADDING
    )
    partial = gl.amd.cdna4.buffer_load(
        PARTIAL,
        offsets + row * LEADER_ROW_PADDING,
        valid[:, :, None],
        0.0,
        cache=".cg" if LOCAL_CORRECTIONS else "",
    ).to(gl.float32)
    if not LOCAL_CORRECTIONS:
        any_residual = _merge_any(flags, REDUCE)
    else:
        any_residual = gl.max(gl.max(flag_words, 0), 0) != 0
    if any_residual:
        residual = gl.amd.cdna4.buffer_load(
            RESIDUAL, offsets, valid[:, :, None] & (flags[:, :, None] & 2 != 0), 0.0
        )
        if not LOCAL_CORRECTIONS:
            partial = gl.where(flags[:, :, None] & 4 != 0, 0.0, partial)
        partial = partial + residual
    numerator = gl.sum(partial * correction[:, :, None], 0)
    denominator = gl.convert_layout(
        denominator, gl.SliceLayout(1, gl.SliceLayout(0, layout)), assert_trivial=True
    )
    reciprocal = gl.div_rn(1.0, gl.where(denominator > 0, denominator, 1.0))
    result = numerator * reciprocal[:, None]
    out_heads = gl.arange(0, 16, gl.SliceLayout(1, gl.SliceLayout(0, layout)))
    gl.store(
        OUTPUT + (row * 16 + out_heads[:, None]) * 512 + values[None, :],
        result.to(gl.bfloat16),
    )


def _allocate_scratch(
    query, splits, record_group, local_corrections, record_padding, leader_row_padding
):
    rows, heads, _ = query.shape
    plane_elements = splits * heads * record_group + record_padding
    residual_elements = rows * (512 // record_group) * plane_elements
    partial_elements = residual_elements + rows * leader_row_padding
    stats_words = 34 if local_corrections else 48
    stats_count = rows * splits * stats_words
    if local_corrections:
        partial_words = partial_elements // 2
        partial_start = stats_count + 1024
        residual_start = partial_start + partial_words + 1024
        arena = torch.empty(
            (residual_start + residual_elements,),
            device=query.device,
            dtype=torch.float32,
        )
        stats = arena[:stats_count]
        partial = arena[partial_start : partial_start + partial_words].view(
            torch.float16
        )
        residual = arena[residual_start:]
    else:
        arena = None
        partial = torch.empty(
            (rows, partial_elements // rows), device=query.device, dtype=torch.float16
        )
        residual = torch.empty(
            (rows, 512 // record_group, plane_elements),
            device=query.device,
            dtype=torch.float32,
        )
        stats = torch.empty((stats_count,), device=query.device, dtype=torch.float32)
    assert residual.numel() * residual.element_size() < 2**31
    assert partial.numel() * partial.element_size() < 2**31
    return (arena, stats, partial, residual)


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
):
    rows, heads, _ = query.shape
    assert heads == 16
    selected = selected_slots.shape[1]
    assert selected == 2048
    assert (kv_cache.shape[0] - 1) * kv_cache.stride(0) + 575 < 2**31
    local_corrections = rows <= 32
    tiles_per_split = 2 if local_corrections else 4
    record_group = 16
    panel = 8
    record_padding = 64 if local_corrections else 0
    leader_row_padding = 128 if local_corrections else 0
    splits = triton.cdiv(selected, tiles_per_split * 64)
    arena, stats, partial, residual = _allocate_scratch(
        query,
        splits,
        record_group,
        local_corrections,
        record_padding,
        leader_row_padding,
    )
    output = torch.empty((rows, heads, 512), device=query.device, dtype=torch.bfloat16)
    row_group = (4 if rows % 4 == 0 else 8) if rows > 32 else 0
    producer_grid = (rows * splits,) if row_group else (rows, splits)
    _attention_partials[producer_grid](
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
        selected,
        splits,
        tiles_per_split,
        residual,
        record_group,
        local_corrections,
        row_group,
        record_padding,
        leader_row_padding,
        ROWS=rows,
        num_warps=4,
    )
    _merge_native[rows, 512 // panel](
        partial,
        stats,
        output,
        residual,
        splits,
        triton.next_power_of_2(splits),
        record_group,
        local_corrections,
        panel,
        record_padding,
        leader_row_padding,
        num_warps=1,
    )
    return output
