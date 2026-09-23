# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m2048.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)


@gluon.jit
def _bitwise_or(a, b):
    return a | b


@gluon.jit
def _row_scan_merge(bits_a, holes_a, bits_b, holes_b):
    return (bits_a | bits_b, holes_a | holes_b)


@gluon.jit
def _first_occupied_tile(bits):
    return gl.inline_asm_elementwise(
        "v_ffbl_b32 $0, $1;",
        constraints="=v,v",
        args=[bits],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _set_priority(priority: gl.constexpr):
    gl.inline_asm_elementwise(
        "s_setprio $1; v_mov_b32 $0, 0;",
        constraints="=v,n",
        args=[priority],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _prime_tile(
    cache,
    current_slots,
    key_smem,
    rotary_smem,
    key_dims,
    key_rotary_dims,
    cache_stride: gl.constexpr,
):
    initial_offset = gl.where(
        current_slots >= 0, current_slots.to(gl.int32) * cache_stride, -2147483648
    )
    initial_offset = gl.multiple_of(initial_offset, 16 if cache_stride % 16 == 0 else 1)
    for part in gl.static_range(4):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            key_smem.slice(part * 128, 128, 1),
            cache,
            initial_offset[:, None]
            + part * 128
            + gl.arange(0, 128, key_dims.type.layout)[None, :],
        )
        if part == 3:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                rotary_smem,
                cache,
                initial_offset[:, None] + 512 + key_rotary_dims[None, :],
            )
        if part == 1 or part == 3:
            gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _row_max(score):
    partial = gl.max(gl.max(gl.reshape(score, (16, 2, 4, 16)), 3), 1)
    exchange = gl.allocate_shared_memory(
        gl.float32, (16, 4), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), partial
    )
    vector_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [4, 1], [0, 1])
    maximum = gl.max(exchange.load(vector_layout), 1)
    return gl.convert_layout(maximum, gl.SliceLayout(1, score.type.layout))


@gluon.jit
def _attention_step(
    cache,
    next_slots,
    active,
    scale,
    resident_queries,
    rotary_query,
    key_smem,
    rotary_smem,
    probability_smem,
    residual_smem,
    key_dims,
    key_rotary_dims,
    maximum,
    denominator_parts,
    numerators,
    cache_stride: gl.constexpr,
    needs_mask: gl.constexpr,
    late_priority: gl.constexpr,
    load_priority: gl.constexpr,
    refill: gl.constexpr,
    early_value: gl.constexpr,
):
    mma: gl.constexpr = numerators[0].type.layout
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    pv_lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 32)
    pv_rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 32)
    residual_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4]
    )
    residual_lhs: gl.constexpr = gl.DotOperandLayout(0, residual_mma, 32)
    residual_rhs: gl.constexpr = gl.DotOperandLayout(1, residual_mma, 32)
    denominator_head_layout: gl.constexpr = gl.SliceLayout(
        1, denominator_parts.type.layout
    )
    _set_priority(load_priority)
    gl.amd.cdna4.async_copy.wait_group(1)
    score = gl.zeros((16, 128), gl.float32, mma)
    for part in gl.static_range(4):
        if part == 2:
            gl.amd.cdna4.async_copy.wait_group(0)
        query_part = resident_queries[part]
        key_part = key_smem.slice(part * 128, 128, 1).permute((1, 0))
        score = gl.amd.cdna4.mfma(
            query_part,
            gl.amd.cdna4.async_copy.load_shared_relaxed(key_part, rhs).to(gl.bfloat16),
            score,
        )
    score = (
        gl.amd.cdna4.mfma(
            rotary_query,
            gl.amd.cdna4.async_copy.load_shared_relaxed(
                rotary_smem.permute((1, 0)), rhs
            ).to(gl.bfloat16),
            score,
        )
        * scale
    )
    if needs_mask:
        mask = gl.convert_layout(active, gl.SliceLayout(0, mma))
        score = gl.where(mask[None, :], score, -float("inf"))
    if early_value:
        first_value = gl.amd.cdna4.async_copy.load_shared_relaxed(
            key_smem.slice(0, 128, 1), pv_rhs
        )
    if not late_priority:
        _set_priority(3)
    tile_maximum = _row_max(score)
    if late_priority:
        _set_priority(3)
    next_max = gl.maximum(maximum, tile_maximum)
    alpha = gl.exp(maximum - next_max)
    probability = gl.exp2((score - next_max[:, None]) * 1.4426950408889634)
    if needs_mask:
        probability = gl.where(
            gl.convert_layout(active, gl.SliceLayout(0, mma))[None, :], probability, 0.0
        )
    scaled_numerators = ()
    for part in gl.static_range(4):
        scaled_numerators += (numerators[part] * alpha[:, None],)
    probability_high = probability.to(gl.float16)
    probability_low = ((probability - probability_high.to(gl.float32)) * 1048576.0).to(
        gl.float8e4nv
    )
    probability_smem.store(probability_high)
    residual_smem.store(probability_low)
    if not early_value:
        first_value = gl.amd.cdna4.async_copy.load_shared_relaxed(
            key_smem.slice(0, 128, 1), pv_rhs
        )
    if refill:
        future_offset = gl.where(
            next_slots >= 0, next_slots.to(gl.int32) * cache_stride, -2147483648
        )
        future_offset = gl.multiple_of(
            future_offset, 16 if cache_stride % 16 == 0 else 1
        )
        value_dims = gl.arange(0, 128, key_dims.type.layout)
    gl.barrier()
    if refill:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            key_smem.slice(0, 128, 1),
            cache,
            future_offset[:, None] + value_dims[None, :],
        )
    local_alpha = gl.convert_layout(alpha, denominator_head_layout)
    tile_mass = gl.sum(gl.reshape(probability, (16, 32, 4)), 2)
    denominator_parts = gl.fma(denominator_parts, local_alpha[:, None], tile_mass)
    leading_probability = probability_smem.load(pv_lhs)
    residual_probability = residual_smem.load(residual_lhs)
    next_numerators = ()
    for part in gl.static_range(4):
        if part == 0:
            value = first_value
        else:
            value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                key_smem.slice(part * 128, 128, 1), pv_rhs
            )
            gl.barrier()
            if refill:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    key_smem.slice(part * 128, 128, 1),
                    cache,
                    future_offset[:, None] + part * 128 + value_dims[None, :],
                )
                if part == 1:
                    gl.amd.cdna4.async_copy.commit_group()
                if part == 3:
                    gl.amd.cdna4.async_copy.buffer_load_to_shared(
                        rotary_smem,
                        cache,
                        future_offset[:, None] + 512 + key_rotary_dims[None, :],
                    )
        numerator = scaled_numerators[part]
        numerator = gl.amd.cdna4.mfma(
            leading_probability, value.to(gl.float16), numerator
        )
        numerator = gl.convert_layout(numerator, residual_mma, assert_trivial=True)
        numerator = gl.amd.cdna4.mfma_scaled(
            residual_probability,
            107,
            "e4m3",
            gl.convert_layout(value, residual_rhs, assert_trivial=True),
            None,
            "e4m3",
            numerator,
        )
        numerator = gl.convert_layout(numerator, mma, assert_trivial=True)
        next_numerators += (numerator,)
    if refill:
        gl.amd.cdna4.async_copy.commit_group()
    return (denominator_parts, next_numerators, next_max)


@gluon.jit
def _attention_mfma(
    query,
    cache,
    selected,
    output,
    scale,
    query_row_stride: gl.constexpr,
    query_head_stride: gl.constexpr,
    cache_stride: gl.constexpr,
    selected_stride: gl.constexpr,
    heads: gl.constexpr,
    selected_count: gl.constexpr,
    rows: gl.constexpr,
    cache_extent: gl.constexpr,
):
    copy_columns: gl.constexpr = 2 if rows == 1024 or rows == 2048 else 1
    publish_together: gl.constexpr = rows == 1024 or rows == 2048
    single_tile: gl.constexpr = selected_count <= 128
    peel_tail: gl.constexpr = rows == 3072 or single_tile
    early_value: gl.constexpr = rows == 3072
    late_priority: gl.constexpr = rows == 1024 or rows == 2048
    if rows == 1024:
        row = rows - 1 - (gl.program_id(0) % 8 * (rows // 8) + gl.program_id(0) // 8)
    elif rows == 2048:
        row = rows - 1 - (gl.program_id(0) % 2 * (rows // 2) + gl.program_id(0) // 2)
    else:
        row = rows - 1 - gl.program_id(0)
    gl.static_assert(heads == 16)
    gl.static_assert(selected_count > 0 and selected_count <= 2048)
    gl.static_assert(cache_extent > 0)
    gl.static_assert(cache_stride >= 0 and cache_stride <= 2147483647)
    gl.static_assert((cache_extent - 1) * cache_stride + 575 < 2147483646)
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    cache_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [32, 2], [4 // copy_columns, copy_columns], [1, 0]
    )
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    head_layout: gl.constexpr = gl.SliceLayout(1, mma)
    if rows > 2048:
        scan_positions = gl.arange(
            0,
            max(128, _artifact_next_power_of_2(selected_count)),
            gl.BlockedLayout([4], [64], [4], [0]),
        )
        scan_slots = gl.load(
            selected + row * selected_stride + scan_positions,
            scan_positions < selected_count,
            -1,
        )
    query_heads = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    query_dims = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
    rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, query_layout))
    query_latent = gl.load(
        query
        + row * query_row_stride
        + query_heads[:, None] * query_head_stride
        + query_dims[None, :]
    )
    query_rotary = gl.load(
        query
        + row * query_row_stride
        + query_heads[:, None] * query_head_stride
        + 512
        + rotary_dims[None, :]
    )
    query_stage = gl.allocate_shared_memory(
        query.dtype.element_ty,
        (16, 512),
        gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
        query_latent,
    )
    if publish_together:
        rotary_stage = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 64),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            query_rotary,
        )
    resident_queries = ()
    for part in gl.static_range(4):
        resident_queries += (query_stage.slice(part * 128, 128, 1).load(lhs),)
    if not publish_together:
        rotary_stage = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 64),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            query_rotary,
        )
    rotary_query = rotary_stage.load(lhs)
    positions = gl.arange(0, 128, gl.SliceLayout(1, cache_layout))
    key_dims = gl.arange(0, 256, gl.SliceLayout(0, cache_layout))
    key_rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, cache_layout))
    maximum = gl.full((16,), -float("inf"), gl.float32, head_layout)
    denominator_parts = gl.sum(
        gl.reshape(gl.zeros((16, 128), gl.float32, mma), (16, 32, 4)), 2
    )
    numerators = ()
    for part in gl.static_range(4):
        numerators += (gl.zeros((16, 128), gl.float32, mma),)
    key_smem = gl.allocate_shared_memory(
        cache.dtype.element_ty,
        (128, 512),
        gl.PaddedSharedLayout(
            [[1024, 16]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [0, 32],
                [0, 64],
                [0, 128],
                [0, 256],
            ],
            [],
            [128, 512],
        ),
    )
    rotary_smem = gl.allocate_shared_memory(
        cache.dtype.element_ty,
        (128, 64),
        gl.PaddedSharedLayout(
            [[1024, 16]],
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
                [0, 32],
            ],
            [],
            [128, 64],
        ),
    )
    probability_smem = gl.allocate_shared_memory(
        gl.float16, (16, 128), gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    )
    residual_smem = gl.allocate_shared_memory(
        gl.float8e4nv, (16, 128), gl.SwizzledSharedLayout(16, 1, 8, [1, 0])
    )
    if rows <= 2048:
        scan_positions = gl.arange(
            0,
            max(128, _artifact_next_power_of_2(selected_count)),
            gl.BlockedLayout([8], [64], [4], [0]),
        )
        scan_slots = gl.load(
            selected + row * selected_stride + scan_positions,
            scan_positions < selected_count,
            -1,
        )
    in_range = scan_positions < selected_count
    if rows > 2048:
        entry_bit = (1 << scan_positions // 128).to(gl.uint32)
        occupied = gl.where(in_range & (scan_slots >= 0), entry_bit, 0)
        holes = gl.where(in_range & (scan_slots < 0), entry_bit, 0)
        summary = gl.reduce(occupied | holes << 16, 0, _bitwise_or)
        tile_bits = summary & 65535
        incomplete_tiles = summary >> 16
        row_is_complete = (incomplete_tiles == 0) & (selected_count % 128 == 0)
    else:
        row_is_complete = (selected_count % 128 == 0) & (
            gl.min(gl.where(in_range, scan_slots, 0), 0) >= 0
        )
        tile_bits = gl.full((), 0, gl.uint32)
        incomplete_tiles = gl.full((), 0, gl.uint32)
        if not row_is_complete:
            entry_bit = (1 << scan_positions // 128).to(gl.uint32)
            tile_bits, incomplete_tiles = gl.reduce(
                (
                    gl.where(in_range & (scan_slots >= 0), entry_bit, 0),
                    gl.where(in_range & (scan_slots < 0), entry_bit, 0),
                ),
                0,
                _row_scan_merge,
            )
    if selected_count % 128 != 0:
        incomplete_tiles |= 1 << selected_count // 128
    _set_priority(1)
    if row_is_complete:
        current_slots = gl.load(selected + row * selected_stride + positions)
        _prime_tile(
            cache,
            current_slots,
            key_smem,
            rotary_smem,
            key_dims,
            key_rotary_dims,
            cache_stride,
        )
        for dense_tile in range(selected_count // 128 - (1 if peel_tail else 0)):
            next_pos = (dense_tile + 1) * 128 + positions
            next_slots = gl.load(
                selected + row * selected_stride + next_pos,
                next_pos < selected_count,
                -1,
            )
            denominator_parts, numerators, maximum = _attention_step(
                cache,
                next_slots,
                None,
                scale,
                resident_queries,
                rotary_query,
                key_smem,
                rotary_smem,
                probability_smem,
                residual_smem,
                key_dims,
                key_rotary_dims,
                maximum,
                denominator_parts,
                numerators,
                cache_stride,
                False,
                late_priority,
                0 if rows == 1024 else 1,
                not single_tile,
                early_value,
            )
        if peel_tail:
            denominator_parts, numerators, maximum = _attention_step(
                cache,
                None,
                None,
                scale,
                resident_queries,
                rotary_query,
                key_smem,
                rotary_smem,
                probability_smem,
                residual_smem,
                key_dims,
                key_rotary_dims,
                maximum,
                denominator_parts,
                numerators,
                cache_stride,
                False,
                late_priority,
                0 if rows == 1024 else 1,
                False,
                early_value,
            )
    elif tile_bits != 0:
        first_pos = _first_occupied_tile(tile_bits) * 128 + positions
        current_slots = gl.load(
            selected + row * selected_stride + first_pos,
            (tile_bits != 0) & (first_pos < selected_count),
            -1,
        )
        _prime_tile(
            cache,
            current_slots,
            key_smem,
            rotary_smem,
            key_dims,
            key_rotary_dims,
            cache_stride,
        )
        while tile_bits != 0:
            current_bit = tile_bits & 0 - tile_bits
            needs_mask = current_bit & incomplete_tiles != 0
            tile_bits = tile_bits & tile_bits - 1
            active = current_slots >= 0
            next_pos = _first_occupied_tile(tile_bits) * 128 + positions
            next_slots = gl.load(
                selected + row * selected_stride + next_pos,
                (tile_bits != 0) & (next_pos < selected_count),
                -1,
            )
            if needs_mask:
                denominator_parts, numerators, maximum = _attention_step(
                    cache,
                    next_slots,
                    active,
                    scale,
                    resident_queries,
                    rotary_query,
                    key_smem,
                    rotary_smem,
                    probability_smem,
                    residual_smem,
                    key_dims,
                    key_rotary_dims,
                    maximum,
                    denominator_parts,
                    numerators,
                    cache_stride,
                    True,
                    late_priority,
                    0 if rows == 1024 else 1,
                    not single_tile,
                    early_value,
                )
            else:
                denominator_parts, numerators, maximum = _attention_step(
                    cache,
                    next_slots,
                    active,
                    scale,
                    resident_queries,
                    rotary_query,
                    key_smem,
                    rotary_smem,
                    probability_smem,
                    residual_smem,
                    key_dims,
                    key_rotary_dims,
                    maximum,
                    denominator_parts,
                    numerators,
                    cache_stride,
                    False,
                    late_priority,
                    0 if rows == 1024 else 1,
                    not single_tile,
                    early_value,
                )
            current_slots = next_slots
    _set_priority(2 if rows == 2048 else 0)
    stage_output: gl.constexpr = rows == 1024 or rows >= 4096
    if rows > 2048 or stage_output:
        gl.amd.cdna4.async_copy.wait_group(0)
    denominator = gl.convert_layout(gl.sum(denominator_parts, 1), head_layout)
    reciprocal = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [2, 2], [1, 0])
    out_heads = gl.arange(0, 16, gl.SliceLayout(1, store_layout))
    if stage_output:
        output_stage = key_smem._reinterpret(
            gl.bfloat16,
            (64, 512),
            gl.PaddedSharedLayout(
                [[512, 8]],
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
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                ],
                [],
                [64, 512],
            ),
        ).slice(0, 16, 0)
        for part in gl.static_range(4):
            result = (numerators[part] * reciprocal[:, None]).to(gl.bfloat16)
            output_stage.slice(part * 128, 128, 1).store(result)
        result = output_stage.load(store_layout)
        out_dims = gl.arange(0, 512, gl.SliceLayout(0, store_layout))
        gl.amd.cdna4.buffer_store(
            result,
            output,
            (row * heads + out_heads[:, None]) * 512 + out_dims[None, :],
            cache=".cs",
        )
    else:
        out_dims = gl.arange(0, 128, gl.SliceLayout(0, store_layout))
        for part in gl.static_range(4):
            result = numerators[part] * reciprocal[:, None]
            result = gl.convert_layout(result.to(gl.bfloat16), store_layout)
            gl.amd.cdna4.buffer_store(
                result,
                output,
                (row * heads + out_heads[:, None]) * 512
                + part * 128
                + out_dims[None, :],
                cache=".cs",
            )
        if rows <= 2048:
            gl.amd.cdna4.async_copy.wait_group(0)
            key_smem._keep_alive()
            rotary_smem._keep_alive()


def sparse_paged_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    softmax_scale: float = 0.0625,
    output: torch.Tensor | None = None,
):
    rows, heads, _ = query.shape
    if output is None:
        output = torch.empty(
            (rows, heads, 512), device=query.device, dtype=torch.bfloat16
        )
    _attention_mfma[rows,](
        query,
        kv_cache,
        selected_slots,
        output,
        softmax_scale,
        query.stride(0),
        query.stride(1),
        kv_cache.stride(0),
        selected_slots.stride(0),
        heads,
        selected_slots.shape[1],
        rows,
        kv_cache.shape[0],
        num_warps=4,
        enable_fp_fusion=rows == 1024,
    )
    return output
