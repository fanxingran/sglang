"""Gluon sparse MLA (DSA ``triton_gluon`` backend) against an FP32 reference."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa import gluon_sparse_mla as gsm
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=600, suite="stage-b-test-1-gpu-small-amd-mi35x")

DIM, D_V, TOPK, SCALE = 576, 512, 2048, 0.0625
UNSUPPORTED = gsm.gluon_sparse_mla_unsupported_reason()


def _inputs(rows, heads, capacity, valid=TOPK, low=0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(rows, heads, DIM, device="cuda", generator=g) * 0.5).to(
        torch.bfloat16
    )
    kv = torch.empty(capacity, 1, DIM, device="cuda", dtype=torch.float8_e4m3fn)
    kv[low:] = (
        torch.randn(capacity - low, 1, DIM, device="cuda", generator=g) * 0.5
    ).to(torch.float8_e4m3fn)
    slots = torch.randint(
        low, capacity, (rows, TOPK), device="cuda", generator=g, dtype=torch.int32
    )
    slots[:, valid:] = -1
    return q, kv, slots.unsqueeze(1)


def _reference(q, kv, indices, check):
    q, idx = q[check].float(), indices[check, 0].long()
    keys = kv[idx.clamp_min(0), 0].float()
    scores = torch.einsum("mhd,mkd->mhk", q, keys) * SCALE
    scores = scores.masked_fill((idx < 0)[:, None, :], float("-inf"))
    return torch.einsum("mhk,mkd->mhd", scores.softmax(-1), keys[..., :D_V])


def _rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@unittest.skipIf(UNSUPPORTED is not None, f"triton_gluon unavailable: {UNSUPPORTED}")
class TestGluonSparseMLA(CustomTestCase):
    def _check(self, fn, rows, heads, capacity=1 << 18, gluon=True, **kw):
        q, kv, indices = _inputs(rows, heads, capacity, **kw)
        out = fn(q[..., :D_V], q[..., D_V:], kv, indices, SCALE, D_V)
        self.assertEqual(tuple(out.shape), (1, rows, heads, D_V))
        check = torch.linspace(0, rows - 1, min(rows, 128), device="cuda").long()
        err = _rel_l2(out[0, check], _reference(q, kv, indices, check))
        # Gluon keeps BF16 Q.K; the Triton fallback quantizes Q and P to FP8.
        self.assertLess(err, 5e-3 if gluon else 5e-2, (rows, heads, kw))
        return out

    def test_decode_rows(self):
        for heads, limit in ((16, 128), (8, 256)):
            for rows in (1, 2, 3, 6, 8, 12, 24, 48, 60, 72, 128, 256):
                with self.subTest(heads=heads, rows=rows):
                    self._check(
                        gsm.gluon_sparse_mla_decode, rows, heads, gluon=rows <= limit
                    )

    def test_partially_valid_slots(self):
        for heads in (8, 16):
            for valid in (1, 700, 2047):
                with self.subTest(heads=heads, valid=valid):
                    self._check(gsm.gluon_sparse_mla_fwd, 12, heads, valid=valid)

    def test_prefill_slices(self):
        # 1024-row slices plus a Triton remainder, and a sub-slice extend.
        for heads in (8, 16):
            for rows in (500, 1024, 3000, 16384 + 1024 + 7):
                with self.subTest(heads=heads, rows=rows):
                    self._check(gsm.gluon_sparse_mla_fwd, rows, heads, gluon=False)

    def test_large_kv_pool(self):
        # > 2**31 bytes of KV: kernels with 32-bit offsets must not be used.
        capacity = 4_000_000
        for heads in (8, 16):
            for rows in (1, 6, 16, 48, 2048):
                with self.subTest(heads=heads, rows=rows):
                    self._check(
                        gsm.gluon_sparse_mla_fwd,
                        rows,
                        heads,
                        capacity=capacity,
                        low=capacity - 400_000,
                        gluon=False,
                    )

    def test_query_layouts(self):
        q, kv, indices = _inputs(6, 16, 1 << 16)
        joined = gsm.gluon_sparse_mla_decode(
            q[..., :D_V], q[..., D_V:], kv, indices, SCALE, D_V
        )
        split = gsm.gluon_sparse_mla_decode(
            q[..., :D_V].contiguous(), q[..., D_V:].contiguous(), kv, indices, SCALE
        )
        fp8 = q.to(torch.float8_e4m3fn)
        from_fp8 = gsm.gluon_sparse_mla_decode(
            fp8[..., :D_V], fp8[..., D_V:], kv, indices, SCALE
        )
        torch.testing.assert_close(joined, split, rtol=0, atol=0)
        check = torch.arange(6, device="cuda")
        ref = _reference(fp8.to(torch.bfloat16), kv, indices, check)
        self.assertLess(_rel_l2(from_fp8[0], ref), 5e-3)

    def test_cuda_graph_replay(self):
        for heads in (8, 16):
            q, kv, indices = _inputs(24, heads, 1 << 16)

            def run():
                return gsm.gluon_sparse_mla_fwd(
                    q[..., :D_V], q[..., D_V:], kv, indices, SCALE
                )

            eager = run()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)


class TestGluonSparseMLAContract(CustomTestCase):
    def test_contract(self):
        ok = dict(
            num_heads=16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            topk=2048,
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        self.assertIsNone(gsm.gluon_sparse_mla_contract_error(**ok))
        for key, value in (
            ("num_heads", 32),
            ("qk_rope_head_dim", 0),
            ("topk", 1024),
            ("kv_cache_dtype", torch.bfloat16),
        ):
            with self.subTest(key=key):
                self.assertIsNotNone(
                    gsm.gluon_sparse_mla_contract_error(**{**ok, key: value})
                )


if __name__ == "__main__":
    unittest.main()
