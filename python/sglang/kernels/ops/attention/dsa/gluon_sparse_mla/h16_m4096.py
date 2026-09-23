# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenAI-Partners/artemis-kernel-integrations@14eb5a6
# src/kernel_packs/gfx950/glm52/sparse_paged_mla/sparse_paged_mla_tp4_m4096.py
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_artifact_next_power_of_2 = triton.constexpr_function(triton.next_power_of_2)


@gluon.jit
def _union_bits(a, b):
    return a | b


@gluon.jit
def _vector_tile_max(score, maximum, mma: gl.constexpr):
    local_max = gl.max(gl.reshape(score, (16, 2, 4, 4, 4)), 4)
    local_max = gl.max(local_max, 1)
    wave_max = gl.max(local_max, 2)
    exchange = gl.allocate_shared_memory(
        gl.float32, (16, 4), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), wave_max
    )
    read_layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [4, 1], [0, 1])
    tile_max = gl.max(exchange.load(read_layout), 1)
    return gl.maximum(maximum, gl.convert_layout(tile_max, gl.SliceLayout(1, mma)))


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
        "s_setprio $1;",
        constraints="=v,n",
        args=[priority],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _prime_cache(
    cache,
    key_smem,
    rotary_smem,
    slots,
    cache_stride: gl.constexpr,
    cache_layout: gl.constexpr,
):
    offset = gl.where(slots >= 0, slots.to(gl.int32) * cache_stride, -2147483648)
    offset = gl.multiple_of(offset, 16 if cache_stride % 16 == 0 else 1)
    dims = gl.arange(0, 256, gl.SliceLayout(0, cache_layout))
    rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, cache_layout))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        key_smem.slice(0, 256, 1), cache, offset[:, None] + dims[None, :]
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        key_smem.slice(256, 256, 1), cache, offset[:, None] + 256 + dims[None, :]
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        rotary_smem, cache, offset[:, None] + 512 + rotary_dims[None, :]
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _step_attention(
    resident_queries,
    cache,
    key_smem,
    rotary_smem,
    probability_high_smem,
    probability_low_smem,
    next_slots,
    active,
    tile_is_incomplete,
    numerators,
    maximum,
    denominator_parts,
    scale,
    cache_stride: gl.constexpr,
    cache_layout: gl.constexpr,
    mma: gl.constexpr,
    rhs: gl.constexpr,
    pv_lhs: gl.constexpr,
    pv_rhs: gl.constexpr,
    fp8_mma: gl.constexpr,
    fp8_lhs: gl.constexpr,
    fp8_rhs: gl.constexpr,
    denominator_head_layout: gl.constexpr,
    HAS_MASK: gl.constexpr,
    rows: gl.constexpr,
):
    _set_priority(1)
    gl.amd.cdna4.async_copy.wait_group(1)
    score = gl.zeros((16, 128), gl.float32, mma)
    for part in gl.static_range(4):
        if part == 2:
            gl.amd.cdna4.async_copy.wait_group(0)
        key_part = key_smem.slice(part * 128, 128, 1).permute((1, 0))
        score = gl.amd.cdna4.mfma(
            resident_queries[part],
            gl.amd.cdna4.async_copy.load_shared_relaxed(key_part, rhs).to(gl.bfloat16),
            score,
        )
    score = (
        gl.amd.cdna4.mfma(
            resident_queries[4],
            gl.amd.cdna4.async_copy.load_shared_relaxed(
                rotary_smem.permute((1, 0)), rhs
            ).to(gl.bfloat16),
            score,
        )
        * scale
    )
    if HAS_MASK:
        if tile_is_incomplete:
            mask = gl.convert_layout(active, gl.SliceLayout(0, mma))
            score = gl.where(mask[None, :], score, -float("inf"))
    if rows != 1024:
        _set_priority(3)
    if rows == 2048:
        next_max = gl.maximum(maximum, gl.max(score, 1))
    else:
        next_max = _vector_tile_max(score, maximum, mma)
    if rows == 1024:
        _set_priority(3)
    alpha = gl.exp(maximum - next_max)
    probability = gl.exp2((score - next_max[:, None]) * 1.4426950408889634)
    if HAS_MASK:
        if tile_is_incomplete:
            mask = gl.convert_layout(active, gl.SliceLayout(0, mma))
            probability = gl.where(mask[None, :], probability, 0.0)
    scaled_numerators = ()
    for part in gl.static_range(4):
        scaled_numerators += (numerators[part] * alpha[:, None],)
    probability_high = probability.to(gl.float16)
    probability_low = ((probability - probability_high.to(gl.float32)) * 1048576.0).to(
        gl.float8e4nv
    )
    probability_high_smem.store(probability_high)
    probability_low_smem.store(probability_low)
    first_value = gl.amd.cdna4.async_copy.load_shared_relaxed(
        key_smem.slice(0, 128, 1), pv_rhs
    )
    future_offset = gl.where(
        next_slots >= 0, next_slots.to(gl.int32) * cache_stride, -2147483648
    )
    future_offset = gl.multiple_of(future_offset, 16 if cache_stride % 16 == 0 else 1)
    value_dims = gl.arange(0, 128, gl.SliceLayout(0, cache_layout))
    rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, cache_layout))
    gl.barrier()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        key_smem.slice(0, 128, 1), cache, future_offset[:, None] + value_dims[None, :]
    )
    local_alpha = gl.convert_layout(alpha, denominator_head_layout)
    tile_mass = gl.sum(gl.reshape(probability, (16, 32, 4)), 2)
    denominator_parts = gl.fma(denominator_parts, local_alpha[:, None], tile_mass)
    high_operand = probability_high_smem.load(pv_lhs)
    low_operand = probability_low_smem.load(fp8_lhs)
    next_numerators = ()
    for part in gl.static_range(4):
        if part == 0:
            value = first_value
        else:
            value = gl.amd.cdna4.async_copy.load_shared_relaxed(
                key_smem.slice(part * 128, 128, 1), pv_rhs
            )
            gl.barrier()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                key_smem.slice(part * 128, 128, 1),
                cache,
                future_offset[:, None] + part * 128 + value_dims[None, :],
            )
            if part == 1:
                gl.amd.cdna4.async_copy.commit_group()
            elif part == 3:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    rotary_smem,
                    cache,
                    future_offset[:, None] + 512 + rotary_dims[None, :],
                )
        numerator = gl.amd.cdna4.mfma(
            high_operand, value.to(gl.float16), scaled_numerators[part]
        )
        numerator = gl.amd.cdna4.mfma_scaled(
            low_operand,
            107,
            "e4m3",
            gl.convert_layout(value, fp8_rhs),
            None,
            "e4m3",
            gl.convert_layout(numerator, fp8_mma),
        )
        next_numerators += (gl.convert_layout(numerator, mma),)
    gl.amd.cdna4.async_copy.commit_group()
    return (next_numerators, next_max, denominator_parts)


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
    program = gl.program_id(0)
    if rows == 1024:
        program = program % 8 * (rows // 8) + program // 8
    elif rows == 2048:
        program = program % 2 * (rows // 2) + program // 2
    row = rows - 1 - program
    gl.static_assert(heads == 16)
    gl.static_assert(selected_count > 0 and selected_count <= 2048)
    gl.static_assert(cache_extent > 0)
    gl.static_assert(cache_stride >= 0 and cache_stride <= 2147483647)
    gl.static_assert((cache_extent - 1) * cache_stride + 575 < 2147483646)
    query_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    cache_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [32, 2], [4, 1] if rows >= 3072 else [2, 2], [1, 0]
    )
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    pv_lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 32)
    pv_rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 32)
    fp8_mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4]
    )
    fp8_lhs: gl.constexpr = gl.DotOperandLayout(0, fp8_mma, 32)
    fp8_rhs: gl.constexpr = gl.DotOperandLayout(1, fp8_mma, 32)
    head_layout: gl.constexpr = gl.SliceLayout(1, mma)
    if rows == 1024 or rows == 2048 or rows >= 3072:
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
    query_heads = gl.arange(0, 16, gl.SliceLayout(1, query_layout))
    query_base = query + row * query_row_stride
    if rows == 1024 or rows == 2048:
        query_dims = gl.arange(0, 1024, gl.SliceLayout(0, query_layout))
        all_query = gl.load(
            query_base + query_heads[:, None] * query_head_stride + query_dims[None, :],
            query_dims[None, :] < 576,
            0,
        )
        all_query_smem = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 1024),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            all_query,
        )
        resident_queries = ()
        for part in gl.static_range(4):
            resident_queries += (all_query_smem.slice(part * 128, 128, 1).load(lhs),)
        resident_queries += (all_query_smem.slice(512, 64, 1).load(lhs),)
        gl.barrier()
    elif rows > 2048:
        query_dims = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
        rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, query_layout))
        latent_query = gl.load(
            query_base + query_heads[:, None] * query_head_stride + query_dims[None, :]
        )
        rotary_query = gl.load(
            query_base
            + query_heads[:, None] * query_head_stride
            + 512
            + rotary_dims[None, :]
        )
        latent_smem = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 512),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            latent_query,
        )
        rotary_q_smem = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 64),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            rotary_query,
        )
        resident_queries = ()
        for part in gl.static_range(4):
            resident_queries += (latent_smem.slice(part * 128, 128, 1).load(lhs),)
        resident_queries += (rotary_q_smem.load(lhs),)
        gl.barrier()
    else:
        query_dims = gl.arange(0, 512, gl.SliceLayout(0, query_layout))
        latent_query = gl.load(
            query_base + query_heads[:, None] * query_head_stride + query_dims[None, :]
        )
        query_latent_smem = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 512),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            latent_query,
        )
        resident_queries = ()
        for part in gl.static_range(4):
            resident_queries += (query_latent_smem.slice(part * 128, 128, 1).load(lhs),)
        gl.barrier()
        rotary_dims = gl.arange(0, 64, gl.SliceLayout(0, query_layout))
        rotary_query = gl.load(
            query_base
            + query_heads[:, None] * query_head_stride
            + 512
            + rotary_dims[None, :]
        )
        query_rotary_smem = gl.allocate_shared_memory(
            query.dtype.element_ty,
            (16, 64),
            gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
            rotary_query,
        )
        resident_queries += (query_rotary_smem.load(lhs),)
        gl.barrier()
    positions = gl.arange(0, 128, gl.SliceLayout(1, cache_layout))
    maximum = gl.full((16,), -float("inf"), gl.float32, head_layout)
    denominator_parts = gl.sum(
        gl.reshape(gl.zeros((16, 128), gl.float32, mma), (16, 32, 4)), 2
    )
    denominator_head_layout: gl.constexpr = gl.SliceLayout(
        1, denominator_parts.type.layout
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
    probability_high_smem = gl.allocate_shared_memory(
        gl.float16, (16, 128), gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    )
    probability_low_smem = gl.allocate_shared_memory(
        gl.float8e4nv, (16, 128), gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    )
    if not (rows == 1024 or rows == 2048 or rows >= 3072):
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
    proof_bits = gl.where(
        scan_slots >= 0,
        (1 << scan_positions // 128).to(gl.uint32),
        (1 << 16 + scan_positions // 128).to(gl.uint32),
    )
    proof = gl.reduce(proof_bits, 0, _union_bits)
    tile_bits = proof & 65535
    incomplete_tiles = proof >> 16
    row_is_complete = (incomplete_tiles == 0) & (selected_count % 128 == 0)
    _set_priority(1)
    if row_is_complete:
        current_slots = gl.load(selected + row * selected_stride + positions)
        _prime_cache(
            cache, key_smem, rotary_smem, current_slots, cache_stride, cache_layout
        )
        for dense_tile in range(selected_count // 128):
            next_pos = (dense_tile + 1) * 128 + positions
            next_slots = gl.load(
                selected + row * selected_stride + next_pos,
                next_pos < selected_count,
                -1,
            )
            numerators, maximum, denominator_parts = _step_attention(
                resident_queries,
                cache,
                key_smem,
                rotary_smem,
                probability_high_smem,
                probability_low_smem,
                next_slots,
                next_slots >= 0,
                False,
                numerators,
                maximum,
                denominator_parts,
                scale,
                cache_stride,
                cache_layout,
                mma,
                rhs,
                pv_lhs,
                pv_rhs,
                fp8_mma,
                fp8_lhs,
                fp8_rhs,
                denominator_head_layout,
                False,
                rows,
            )
    else:
        first_pos = _first_occupied_tile(tile_bits) * 128 + positions
        current_slots = gl.load(
            selected + row * selected_stride + first_pos,
            (tile_bits != 0) & (first_pos < selected_count),
            -1,
        )
        _prime_cache(
            cache, key_smem, rotary_smem, current_slots, cache_stride, cache_layout
        )
        while tile_bits != 0:
            tile_bit = tile_bits & 0 - tile_bits
            tile_is_incomplete = incomplete_tiles & tile_bit != 0
            tile_bits = tile_bits & tile_bits - 1
            active = current_slots >= 0
            next_pos = _first_occupied_tile(tile_bits) * 128 + positions
            next_slots = gl.load(
                selected + row * selected_stride + next_pos,
                (tile_bits != 0) & (next_pos < selected_count),
                -1,
            )
            numerators, maximum, denominator_parts = _step_attention(
                resident_queries,
                cache,
                key_smem,
                rotary_smem,
                probability_high_smem,
                probability_low_smem,
                next_slots,
                active,
                tile_is_incomplete,
                numerators,
                maximum,
                denominator_parts,
                scale,
                cache_stride,
                cache_layout,
                mma,
                rhs,
                pv_lhs,
                pv_rhs,
                fp8_mma,
                fp8_lhs,
                fp8_rhs,
                denominator_head_layout,
                True,
                rows,
            )
            current_slots = next_slots
    _set_priority(0)
    if rows != 2048:
        gl.amd.cdna4.async_copy.wait_group(0)
    denominator = gl.convert_layout(gl.sum(denominator_parts, 1), head_layout)
    reciprocal = 1.0 / gl.where(denominator > 0, denominator, 1.0)
    if rows == 2048:
        output_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [2, 2], [1, 0])
        out_heads = gl.arange(0, 16, gl.SliceLayout(1, output_layout))
        out_dims = gl.arange(0, 128, gl.SliceLayout(0, output_layout))
        for part in gl.static_range(4):
            result = (numerators[part] * reciprocal[:, None]).to(gl.bfloat16)
            result = gl.convert_layout(result, output_layout)
            gl.store(
                output
                + (row * heads + out_heads[:, None]) * 512
                + part * 128
                + out_dims[None, :],
                result,
                cache_modifier=".cs",
            )
        gl.amd.cdna4.async_copy.wait_group(0)
        key_smem._keep_alive()
        rotary_smem._keep_alive()
    else:
        output_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        results = ()
        for part in gl.static_range(4):
            results += ((numerators[part] * reciprocal[:, None]).to(gl.bfloat16),)
        low = gl.reshape(
            gl.permute(gl.join(results[0], results[1]), (0, 2, 1)), (16, 256)
        )
        high = gl.reshape(
            gl.permute(gl.join(results[2], results[3]), (0, 2, 1)), (16, 256)
        )
        result = gl.reshape(gl.permute(gl.join(low, high), (0, 2, 1)), (16, 512))
        result = gl.convert_layout(result, output_layout)
        out_heads = gl.arange(0, 16, gl.SliceLayout(1, output_layout))
        out_dims = gl.arange(0, 512, gl.SliceLayout(0, output_layout))
        gl.store(
            output + (row * heads + out_heads[:, None]) * 512 + out_dims[None, :],
            result,
            cache_modifier=".cs",
        )


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
        enable_fp_fusion=False,
    )
    return output
