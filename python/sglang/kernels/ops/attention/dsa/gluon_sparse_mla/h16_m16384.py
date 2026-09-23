# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m16384.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)


@gluon.jit
def _score_maximum(scores, staged, partial_layout: gl.constexpr):
    quartets = gl.convert_layout(
        gl.max(gl.reshape(scores, (16, 8, 4)), 2), partial_layout
    )
    if staged.shape[1] == 4:
        paired = gl.max(gl.reshape(quartets, (16, 2, 2, 2)), 2)
        staged.store(gl.reshape(paired, (16, 4)))
    else:
        staged.store(quartets)
    replicated_layout: gl.constexpr = gl.BlockedLayout(
        [1, staged.shape[1]], [16, 4], [2, 1], [0, 1]
    )
    maximum = gl.max(staged.load(replicated_layout), 1)
    return gl.convert_layout(maximum, gl.SliceLayout(1, scores.type.layout))


@gluon.jit
def _wave_has_value(values, THRESHOLD: gl.constexpr):
    wave_mask = gl.inline_asm_elementwise(
        "v_cmp_ge_u32_e64 $0, $1, $2",
        "=s,v,s",
        [values, THRESHOLD],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    flat_mask = gl.reshape(gl.permute(wave_mask, (1, 0)), (64,))
    lane_zero = gl.full((1,), 0, gl.int32, flat_mask.type.layout)
    return gl.sum(gl.gather(flat_mask, lane_zero, 0), 0) != 0


@gluon.jit
def _packed_basis_vote(encodings):
    pairs = gl.permute(gl.reshape(encodings, (16, 4, 2, 2, 2)), (0, 1, 4, 2, 3))
    even, odd = gl.split(pairs)
    first, third = gl.split(even)
    second, fourth = gl.split(odd)
    possible = gl.inline_asm_elementwise(
        "v_or3_b32 $0, $1, $2, $3\nv_bitop3_b32 $0, $0, $4, $5 bitop3:0xa8",
        "=&v,v,v,v,v,s",
        [
            first,
            second,
            third,
            fourth,
            gl.full(first.shape, 49152, gl.uint16, first.type.layout),
        ],
        dtype=gl.uint16,
        is_pure=True,
        pack=2,
    )
    vote = False
    if _wave_has_value(gl.max(possible.to(gl.uint32), 2), 1):
        largest = gl.inline_asm_elementwise(
            "v_pk_max_u16 $0, $1, $2\nv_pk_max_u16 $0, $0, $3\nv_pk_max_u16 $0, $0, $4",
            "=&v,v,v,v,v",
            [first, second, third, fourth],
            dtype=gl.uint16,
            is_pure=True,
            pack=2,
        )
        vote = _wave_has_value(gl.max(largest.to(gl.uint32), 2), 16429)
    return vote


@gluon.jit
def _basis_vote(weights, CACHE_PACKED: gl.constexpr):
    vote_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [2, 1], [0, 1])
    vote_weights = gl.convert_layout(weights, vote_layout, assert_trivial=True)
    encodings = vote_weights.to(gl.uint16, bitcast=True)
    if CACHE_PACKED:
        return _packed_basis_vote(encodings)
    else:
        largest = gl.max(gl.reshape(encodings.to(gl.uint32), (16, 4, 8)), 2)
        return _wave_has_value(largest, 16429)


@gluon.jit
def _rounded_scores(scores):
    return gl.inline_asm_elementwise(
        "", "=v,0", [scores], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _tile_probability(
    scores, shift, valid, word, FULL_TILE: gl.constexpr, NONEMPTY: gl.constexpr
):
    if FULL_TILE or word == 4294967295:
        probability = gl.exp2((scores - shift[:, None]) * 1.4426950408889634)
    else:
        if NONEMPTY:
            safe_shift = shift
        else:
            safe_shift = gl.where(word != 0, shift, 0.0)
        probability = gl.where(
            valid[None, :],
            gl.exp2((scores - safe_shift[:, None]) * 1.4426950408889634),
            0.0,
        )
    return probability


@gluon.jit
def _rescale_accumulators(delta, denominator, numerator):
    correction = gl.exp2(delta * 1.4426950408889634)
    partial_correction = gl.convert_layout(
        correction, gl.SliceLayout(1, denominator.type.layout)
    )
    return (denominator * partial_correction[:, None], numerator * correction[:, None])


@gluon.jit
def _attention_tile(
    context,
    state,
    block,
    next_block,
    word,
    FULL_TILE: gl.constexpr,
    INITIAL_TILE: gl.constexpr,
    CACHE_PACKED: gl.constexpr,
    EARLY_SUM: gl.constexpr,
    PRIORITIZE_KEYS: gl.constexpr,
):
    (
        Cache,
        Rotary,
        query_low,
        query_high,
        query_rotary,
        staged_slots,
        maximum_stage,
        selected_column,
        latent_column,
        rotary_selected,
        rotary_column,
        softmax_scale,
    ) = context
    shift, denominator, numerator, pending_slot = state
    mma: gl.constexpr = numerator.type.layout
    query_operand: gl.constexpr = query_low.type.layout
    cache_operand: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    denominator_layout: gl.constexpr = denominator.type.layout
    tile_count: gl.constexpr = staged_slots.shape[0] // 32
    rotary_slot = staged_slots.gather(block * 32 + rotary_selected, 0)
    if CACHE_PACKED:
        latent_slot = pending_slot * 8
        latent_offsets = latent_slot[:, None] + latent_column[None, :]
    else:
        latent_offsets = pending_slot[:, None] + latent_column[None, :]
    if CACHE_PACKED:
        rotary_offsets = rotary_slot[:, None] + rotary_column[None, :]
    else:
        rotary_offsets = rotary_slot[:, None] + 512 + rotary_column[None, :]
    latent_offsets = gl.max_contiguous(gl.multiple_of(latent_offsets, [1, 16]), [1, 16])
    rotary_offsets = gl.max_contiguous(gl.multiple_of(rotary_offsets, [1, 8]), [1, 8])
    cache_latent = gl.amd.cdna4.buffer_load(Cache, latent_offsets)
    if CACHE_PACKED:
        cache_rotary = gl.amd.cdna4.buffer_load(Rotary, rotary_offsets)
    else:
        cache_rotary = gl.amd.cdna4.buffer_load(Cache, rotary_offsets)
    staged_latent = gl.allocate_shared_memory(
        cache_latent.dtype,
        (32, 512),
        gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0]),
        cache_latent,
    )
    if PRIORITIZE_KEYS:
        gl.inline_asm_elementwise(
            "s_setprio 1", "=s", [], gl.int32, is_pure=False, pack=1
        )
    keys_view = staged_latent.permute((1, 0))
    scores = gl.full((16, 32), 0.0, gl.float32, mma)
    keys_low = keys_view.slice(0, 256, 0).load(cache_operand).to(gl.bfloat16)
    scores = gl.amd.cdna4.mfma(query_low, keys_low, scores)
    keys_high = keys_view.slice(256, 256, 0).load(cache_operand).to(gl.bfloat16)
    scores = gl.amd.cdna4.mfma(query_high, keys_high, scores)
    keys_rotary = gl.convert_layout(cache_rotary.T, cache_operand).to(gl.bfloat16)
    scores = gl.amd.cdna4.mfma(query_rotary, keys_rotary, scores) * softmax_scale
    if PRIORITIZE_KEYS:
        gl.inline_asm_elementwise(
            "s_setprio 0", "=s", [], gl.int32, is_pure=False, pack=1
        )
    score_column = gl.arange(0, 32, layout=gl.SliceLayout(0, mma))
    valid = gl.cast(word, gl.uint32) >> score_column & 1 != 0
    if INITIAL_TILE:
        if CACHE_PACKED:
            scores = _rounded_scores(scores)
        next_shift = _score_maximum(scores, maximum_stage, denominator_layout)
        probability = gl.exp2((scores - next_shift[:, None]) * 1.4426950408889634)
    else:
        probability = _tile_probability(
            scores, shift, valid, word, FULL_TILE, EARLY_SUM
        )
    if EARLY_SUM:
        probability_sum = gl.convert_layout(
            gl.sum(gl.reshape(probability, (16, 8, 4)), 2), denominator_layout
        )
    weight_stage = gl.allocate_shared_memory(
        gl.bfloat16,
        (16, 32),
        gl.SwizzledSharedLayout(8, 2, 4, [1, 0]),
        probability.to(gl.bfloat16),
    )
    weights = weight_stage.load(query_operand)
    if not INITIAL_TILE:
        next_shift = shift
        if _basis_vote(weights, CACHE_PACKED):
            if CACHE_PACKED:
                scores = _rounded_scores(scores)
            if not FULL_TILE:
                scores = gl.where(valid[None, :], scores, -float("inf"))
            tile_shift = _score_maximum(scores, maximum_stage, denominator_layout)
            next_shift = gl.maximum(shift, tile_shift)
            correction_delta = shift - next_shift
            probability = _tile_probability(
                scores, next_shift, valid, word, FULL_TILE, EARLY_SUM
            )
            if EARLY_SUM:
                probability_sum = gl.convert_layout(
                    gl.sum(gl.reshape(probability, (16, 8, 4)), 2), denominator_layout
                )
            weight_stage.store(probability.to(gl.bfloat16))
            weights = weight_stage.load(query_operand)
            denominator, numerator = _rescale_accumulators(
                correction_delta, denominator, numerator
            )
    if not EARLY_SUM:
        probability_sum = gl.convert_layout(
            gl.sum(gl.reshape(probability, (16, 8, 4)), 2), denominator_layout
        )
    if FULL_TILE:
        next_position = (block + 1) % tile_count * 32 + selected_column
    else:
        next_position = next_block * 32 + selected_column
    pending_slot = staged_slots.gather(next_position, 0)
    denominator = denominator + probability_sum
    gl.inline_asm_elementwise(
        "s_setprio 2" if CACHE_PACKED else "s_setprio 1",
        "=s",
        [],
        gl.int32,
        is_pure=False,
        pack=1,
    )
    values = staged_latent.load(cache_operand).to(gl.bfloat16)
    numerator = gl.amd.cdna4.mfma(weights, values, numerator)
    gl.inline_asm_elementwise("s_setprio 0", "=s", [], gl.int32, is_pure=False, pack=1)
    return (next_shift, denominator, numerator, pending_slot)


@gluon.jit
def _first_set_bit(mask):
    return gl.inline_asm_elementwise(
        "s_ff1_i32_b64 $0, $1", "=s,s", [mask], dtype=gl.int32, is_pure=True, pack=1
    )


@gluon.jit
def _stage_row_slots(
    row,
    Slots,
    SLOT_ROW_STRIDE: gl.constexpr,
    CACHE_ROW_STRIDE: gl.constexpr,
    SELECTED: gl.constexpr,
    CACHE_PACKED: gl.constexpr,
):
    tile_count: gl.constexpr = triton.cdiv(_artifact_next_power_of_2(SELECTED), 32)
    scan_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [2, 1], [1, 0])
    flat_layout: gl.constexpr = gl.BlockedLayout([4], [64], [2], [0])
    scan_position = gl.arange(0, tile_count * 32, layout=flat_layout)
    all_slots = gl.load(
        Slots + row * SLOT_ROW_STRIDE + scan_position, scan_position < SELECTED, -1
    )
    if CACHE_PACKED:
        staged_indices = gl.where(all_slots >= 0, all_slots.to(gl.int32) * 64, -64)
    else:
        staged_indices = gl.where(
            all_slots >= 0, all_slots.to(gl.int32) * CACHE_ROW_STRIDE, -2147483648
        )
    staged_slots = gl.allocate_shared_memory(
        staged_indices.dtype,
        (tile_count * 32,),
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        staged_indices,
    )
    column = gl.arange(0, 32, layout=gl.SliceLayout(0, scan_layout))
    present = gl.reshape(all_slots >= 0, (tile_count, 32))
    validity_word = gl.sum(
        gl.where(present, gl.cast(1, gl.uint32) << column[None, :], 0), 1
    )
    staged_validity = gl.allocate_shared_memory(
        gl.uint32, (tile_count,), gl.SwizzledSharedLayout(1, 1, 1, [0]), validity_word
    )
    bounds_layout: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
    words = staged_validity.load(bounds_layout)
    active_ballot = gl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, 0, $1",
        "=s,v",
        [words],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    incomplete_ballot = gl.inline_asm_elementwise(
        "v_cmp_ne_u32_e64 $0, -1, $1",
        "=s,v",
        [words],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    first_word = gl.full((1,), 0, gl.int32, bounds_layout)
    active_mask = gl.sum(gl.gather(active_ballot, first_word, 0), 0)
    incomplete_mask = gl.sum(gl.gather(incomplete_ballot, first_word, 0), 0)
    if tile_count < 64:
        active_mask = active_mask & (1 << tile_count) - 1
        incomplete_mask = incomplete_mask & (1 << tile_count) - 1
    first_incomplete = incomplete_mask & 0 - incomplete_mask
    full_prefix = gl.where(
        incomplete_mask != 0,
        63 - gl.extra.libdevice.clz(first_incomplete.to(gl.int64)),
        tile_count,
    )
    return (staged_slots, staged_validity, active_mask, first_incomplete, full_prefix)


@gluon.jit
def _load_row_query(
    row,
    Query,
    QUERY_ROW_STRIDE: gl.constexpr,
    QUERY_HEAD_STRIDE: gl.constexpr,
    QUERY_CG: gl.constexpr,
    query_operand: gl.constexpr,
):
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 1], [1, 0])
    head = gl.arange(0, 16, layout=gl.SliceLayout(1, query_layout))
    latent = gl.arange(0, 512, layout=gl.SliceLayout(0, query_layout))
    query_base = Query + row * QUERY_ROW_STRIDE + head[:, None] * QUERY_HEAD_STRIDE
    query_cache: gl.constexpr = ".cg" if QUERY_CG else ""
    query = gl.load(query_base + latent[None, :], cache_modifier=query_cache)
    rotary_query_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [16, 4], [2, 1], [0, 1]
    )
    rotary_head = gl.arange(0, 16, layout=gl.SliceLayout(1, rotary_query_layout))
    rotary = gl.arange(0, 64, layout=gl.SliceLayout(0, rotary_query_layout))
    rotary_query_base = (
        Query + row * QUERY_ROW_STRIDE + rotary_head[:, None] * QUERY_HEAD_STRIDE
    )
    query_rotary = gl.load(
        rotary_query_base + 512 + rotary[None, :], cache_modifier=query_cache
    )
    query_stage = gl.allocate_shared_memory(
        query.dtype, (16, 512), gl.SwizzledSharedLayout(8, 1, 8, [1, 0]), query
    )
    query = query_stage.load(query_operand)
    query_rotary = gl.convert_layout(query_rotary, query_operand, assert_trivial=True)
    query_low, query_high = gl.split(
        gl.permute(gl.reshape(query, (16, 2, 256)), (0, 2, 1))
    )
    query_low = gl.convert_layout(query_low, query_operand, assert_trivial=True)
    query_high = gl.convert_layout(query_high, query_operand, assert_trivial=True)
    return (query_low, query_high, query_rotary)


@gluon.jit
def _store_row(row, Output, denominator, numerator):
    mma: gl.constexpr = numerator.type.layout
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 1], [1, 0])
    denominator = gl.convert_layout(gl.sum(denominator, 1), gl.SliceLayout(1, mma))
    inverse = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    result = numerator * inverse[:, None]
    output_stage = gl.allocate_shared_memory(
        gl.bfloat16,
        (16, 512),
        gl.SwizzledSharedLayout(8, 1, 8, [1, 0]),
        result.to(gl.bfloat16),
    )
    result = output_stage.load(query_layout)
    output_head = gl.arange(0, 16, layout=gl.SliceLayout(1, query_layout))
    output_column = gl.arange(0, 512, layout=gl.SliceLayout(0, query_layout))
    gl.store(
        Output + (row * 16 + output_head[:, None]) * 512 + output_column[None, :],
        result,
        cache_modifier=".cs",
    )


@gluon.jit
def _attention_row(
    row,
    Query,
    Cache,
    Rotary,
    Slots,
    Output,
    softmax_scale,
    QUERY_ROW_STRIDE: gl.constexpr,
    QUERY_HEAD_STRIDE: gl.constexpr,
    CACHE_ROW_STRIDE: gl.constexpr,
    SLOT_ROW_STRIDE: gl.constexpr,
    SELECTED: gl.constexpr,
    QUERY_CG: gl.constexpr,
    CACHE_PACKED: gl.constexpr,
    EMPTY_BYPASS: gl.constexpr,
    EARLY_SUM: gl.constexpr,
    PRIORITIZE_KEYS: gl.constexpr,
    MAXIMUM_PARTS: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 2]
    )
    query_operand: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    cache_layout: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [2, 1], [1, 0])
    staged_slots, staged_validity, active_mask, first_incomplete, full_prefix = (
        _stage_row_slots(
            row, Slots, SLOT_ROW_STRIDE, CACHE_ROW_STRIDE, SELECTED, CACHE_PACKED
        )
    )
    tile_count: gl.constexpr = staged_slots.shape[0] // 32
    if EMPTY_BYPASS and active_mask == 0:
        empty_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 1], [1, 0])
        empty_head = gl.arange(0, 16, gl.SliceLayout(1, empty_layout))
        empty_column = gl.arange(0, 512, gl.SliceLayout(0, empty_layout))
        gl.store(
            Output + (row * 16 + empty_head[:, None]) * 512 + empty_column[None, :],
            gl.full((16, 512), 0, gl.bfloat16, empty_layout),
            cache_modifier=".cs",
        )
    else:
        query_low, query_high, query_rotary = _load_row_query(
            row, Query, QUERY_ROW_STRIDE, QUERY_HEAD_STRIDE, QUERY_CG, query_operand
        )
        selected_column = gl.arange(0, 32, layout=gl.SliceLayout(1, cache_layout))
        latent_column = gl.arange(0, 512, layout=gl.SliceLayout(0, cache_layout))
        rotary_layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [2, 1], [0, 1])
        rotary_selected = gl.arange(0, 32, layout=gl.SliceLayout(1, rotary_layout))
        rotary_column = gl.arange(0, 64, layout=gl.SliceLayout(0, rotary_layout))
        shift = gl.full((16,), -float("inf"), gl.float32, gl.SliceLayout(1, mma))
        denominator_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[],
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
            warp_bases=[[0, 4]],
            block_bases=[],
            shape=[16, 8],
        )
        denominator = gl.full((16, 8), 0.0, gl.float32, denominator_layout)
        numerator = gl.full((16, 512), 0.0, gl.float32, mma)
        maximum_stage = gl.allocate_shared_memory(
            gl.float32, (16, MAXIMUM_PARTS), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        pending_slot = staged_slots.gather(selected_column, 0)
        context = (
            Cache,
            Rotary,
            query_low,
            query_high,
            query_rotary,
            staged_slots,
            maximum_stage,
            selected_column,
            latent_column,
            rotary_selected,
            rotary_column,
            softmax_scale,
        )
        state = (shift, denominator, numerator, pending_slot)
        first_block = 0
        if full_prefix > 0:
            state = _attention_tile(
                context,
                state,
                0,
                0,
                4294967295,
                True,
                True,
                CACHE_PACKED,
                EARLY_SUM,
                PRIORITIZE_KEYS,
            )
            first_block = 1
        for block in range(first_block, full_prefix):
            state = _attention_tile(
                context,
                state,
                block,
                0,
                4294967295,
                True,
                False,
                CACHE_PACKED,
                EARLY_SUM,
                PRIORITIZE_KEYS,
            )
        remaining = active_mask & 0 - first_incomplete
        block = _first_set_bit(remaining) & tile_count - 1
        pending_slot = staged_slots.gather(block * 32 + selected_column, 0)
        state = (state[0], state[1], state[2], pending_slot)
        while remaining != 0:
            remaining = remaining & remaining - 1
            next_block = _first_set_bit(remaining) & tile_count - 1
            validity_index = block + gl.arange(
                0, 1, layout=gl.BlockedLayout([1], [64], [2], [0])
            )
            word = gl.sum(staged_validity.gather(validity_index, 0), 0)
            state = _attention_tile(
                context,
                state,
                block,
                next_block,
                word,
                False,
                False,
                CACHE_PACKED,
                EARLY_SUM,
                PRIORITIZE_KEYS,
            )
            block = next_block
        _store_row(row, Output, state[1], state[2])


@gluon.jit
def _persistent_attention(
    Query,
    Cache,
    Rotary,
    Slots,
    Output,
    Counter,
    softmax_scale,
    ROWS: gl.constexpr,
    QUERY_ROW_STRIDE: gl.constexpr,
    QUERY_HEAD_STRIDE: gl.constexpr,
    CACHE_ROW_STRIDE: gl.constexpr,
    SLOT_ROW_STRIDE: gl.constexpr,
    SELECTED: gl.constexpr,
    QUERY_CG: gl.constexpr,
    CACHE_PACKED: gl.constexpr,
):
    prioritize_keys: gl.constexpr = CACHE_PACKED and (ROWS < 6144 or ROWS >= 12288)
    early_sum: gl.constexpr = ROWS >= 6144 and (not CACHE_PACKED or ROWS < 12288)
    maximum_parts: gl.constexpr = 8 if CACHE_PACKED and ROWS >= 12288 else 4
    ticket = gl.program_id(0)
    while ticket < ROWS:
        _attention_row(
            ROWS - 1 - ticket,
            Query,
            Cache,
            Rotary,
            Slots,
            Output,
            softmax_scale,
            QUERY_ROW_STRIDE,
            QUERY_HEAD_STRIDE,
            CACHE_ROW_STRIDE,
            SLOT_ROW_STRIDE,
            SELECTED,
            QUERY_CG,
            CACHE_PACKED,
            ROWS >= 6144 and (not CACHE_PACKED),
            early_sum,
            prioritize_keys,
            maximum_parts,
        )
        ticket = gl.atomic_add(Counter, 1, sem="relaxed", scope="gpu")


@gluon.jit
def _reset_queue(Counter, FIRST_TICKET: gl.constexpr):
    gl.store(Counter, FIRST_TICKET)


@gluon.jit
def _pack_cache(
    Cache,
    Latent,
    Rotary,
    Counter,
    ROWS: gl.constexpr,
    STRIDE: gl.constexpr,
    FIRST_TICKET: gl.constexpr,
):
    if gl.program_id(0) == 0:
        gl.store(Counter, FIRST_TICKET)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    row = gl.program_id(0) * 64 + gl.arange(0, 64, gl.SliceLayout(1, layout))
    column = gl.arange(0, 64, gl.SliceLayout(0, layout))
    rotary = gl.load(
        Cache + row[:, None] * STRIDE + 512 + column[None, :], row[:, None] < ROWS, 0.0
    )
    gl.store(
        Rotary + row[:, None] * 64 + column[None, :],
        rotary.to(gl.bfloat16),
        row[:, None] < ROWS,
    )
    latent_column = gl.arange(0, 512, gl.SliceLayout(0, layout))
    latent = gl.load(
        Cache + row[:, None] * STRIDE + latent_column[None, :], row[:, None] < ROWS, 0.0
    )
    gl.store(
        Latent + row[:, None] * 512 + latent_column[None, :],
        latent,
        row[:, None] < ROWS,
    )


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
    output: torch.Tensor | None = None,
):
    rows, heads, channels = query.shape
    assert rows > 0 and heads == 16 and (channels == 576)
    assert query.dtype == torch.bfloat16
    assert kv_cache.ndim == 2 and kv_cache.shape[1] == 576
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert selected_slots.ndim == 2 and selected_slots.shape[0] == rows
    assert 0 < selected_slots.shape[1] <= 2048
    assert selected_slots.dtype in (torch.int32, torch.int64)
    assert query.stride(-1) == kv_cache.stride(-1) == selected_slots.stride(-1) == 1
    assert query.device == kv_cache.device == selected_slots.device
    assert softmax_scale > 0
    if output is None:
        output = torch.empty(
            (rows, heads, 512), device=query.device, dtype=torch.bfloat16
        )
    assert (
        0 < kv_cache.stride(0)
        and kv_cache.stride(0) % 16 == 0
        and ((kv_cache.shape[0] - 1) * kv_cache.stride(0) + 576 < 2**31 - 1024)
    )
    workers = min(rows, 1024)
    counter = torch.empty((), device=query.device, dtype=torch.int32)
    cache = kv_cache
    cache_stride = kv_cache.stride(0)
    cache_packed = False
    rotary = kv_cache
    if (
        rows >= 4096
        and 0 < kv_cache.shape[0] <= 2 * rows
        and (512 <= cache_stride < 2**31)
    ):
        cache = torch.empty(
            (kv_cache.shape[0], 512), device=kv_cache.device, dtype=kv_cache.dtype
        )
        rotary = torch.empty(
            (kv_cache.shape[0], 64), device=kv_cache.device, dtype=torch.bfloat16
        )
        _pack_cache[triton.cdiv(kv_cache.shape[0], 64),](
            kv_cache,
            cache,
            rotary,
            counter,
            kv_cache.shape[0],
            cache_stride,
            workers,
            num_warps=4,
        )
        cache_packed = True
        cache_stride = 512
    else:
        _reset_queue[1,](counter, workers, num_warps=1)
    arguments = (
        query.stride(0),
        query.stride(1),
        cache_stride,
        selected_slots.stride(0),
        selected_slots.shape[1],
        rows >= 12288,
        cache_packed,
    )
    _persistent_attention[workers,](
        query,
        cache,
        rotary,
        selected_slots,
        output,
        counter,
        softmax_scale,
        rows,
        *arguments,
        num_warps=2,
        enable_fp_fusion=cache_packed,
    )
    return output
