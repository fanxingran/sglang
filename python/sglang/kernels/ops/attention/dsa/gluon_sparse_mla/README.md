# triton_gluon DSA sparse MLA（GLM-5.2，MI355X）

跟踪分支，不向上游提交。分支 `amd-dsa-triton-gluon`，commit `393d7ba9fa`，仓库 `https://github.com/fanxingran/sglang`。

在 SGLang 里加了一个可选的 DSA backend `triton_gluon`。Kernel 来自 Artemis PR 11（`OpenAI-Partners/artemis-kernel-integrations@14eb5a6`）的 GLM-5.2 sparse paged MLA，按 head 数和行数原样放进本目录。默认 backend 仍是 `triton`。AgentX 官方口径的 P90 Interactivity：TP4/EP4/C8 是 +1.5%，TP8/EP1/C1 是 −3.5%。不是两个格子都有正收益，见文末结论。这套代码只留在本分支供以后对照。

## 代码在哪

| 路径 | 作用 |
|---|---|
| `gluon_sparse_mla/h{8,16}_m*.py` | Gluon kernel。`h16` 是每卡 16 head（TP4），`h8` 是每卡 8 head（TP8）。`m` 是该文件固定的行数。`@gluon.jit` 函数相对 Artemis 没有改动 |
| `gluon_sparse_mla/__init__.py` | Dispatcher：按行数选 kernel、pad、把长 prefill 切段、KV 跨度 ≥ 2 GiB 时跳过 32 位偏移的 kernel |
| `srt/layers/attention/dsa_backend.py` | `forward_extend` 走 `gluon_sparse_mla_fwd`，`forward_decode` 走 `gluon_sparse_mla_decode` |
| `srt/arg_groups/fields/exec_.py` | `--dsa-prefill-backend` / `--dsa-decode-backend` 增加 `triton_gluon` |
| `srt/arg_groups/overrides.py` | 显式选择时，环境不满足直接 `ValueError` |
| `srt/models/.../forward_mla_rocm.py` | gluon 这一步的 query 保持 BF16。Verify 和 draft extend 看 decode backend |
| `srt/mem_cache/kv_cache_configurator.py` | HIP 上使用原始 FP8 KV 布局（512 latent + 64 RoPE），不加 per-block scale |
| `test/registered/kernels/ops/attention/test_gluon_sparse_mla.py` | 正确性、大 KV、CUDA graph |
| `test/registered/unit/test_model_overrides.py` | 门控 |

6 个 prefill 文件只在最外层 `sparse_paged_mla` 上多了可选参数 `output`，用来把一段 prefill 直接写进调用方的 buffer。`tp4_m8192` 没有搬：它每次调用会重打包整个 KV pool。

几何：query BF16 `[M, H, 576]`，KV FP8 E4M3 `[tokens, 576]`，top-k 2048，`sm_scale = 0.0625`，输出 BF16 `[M, H, 512]`。`H` 只能是 8 或 16。

## 环境

记录这次数据的环境：

```text
GPU:     MI355X (gfx950)
节点:    pit2-p03-g53
镜像:    lmsysorg/sglang-rocm:v0.5.20-rocm10-mi35x-20260922
容器:    fanxingran_glm5.2-sglang-rocm10
torch:   2.11.0+rocm10.0.0
triton:  3.8.0
模型:    /share_nfs/fanxingran/models/GLM-5.2-MXFP4
SGLang:  本分支，PYTHONPATH 指向工作树的 python/，没有装进镜像
```

门控（`gluon_sparse_mla_unsupported_reason`）：必须是 ROCm、当前 GPU 的 `gcnArchName` 为 `gfx950`、Triton ≥ 3.8.0，且存在 `triton.experimental.gluon.language.amd.cdna4`。KV cache 必须是 `fp8_e4m3`。不满足时，显式传入 `triton_gluon` 会在启动时报 `ValueError`，不会静默退回。ROCm 7.2 镜像自带 Triton 3.7，这批 kernel 编不过，不要在那张镜像上开这个 backend。

两边 arm 用的是同一张 ROCm 10 镜像。Kernel 对比里没有 ROCm 7 和 ROCm 10 的差异。

## 怎么打开

Server 参数，其余与 InferenceX `glm5.2_fp4_mi355x_sglang_mtp.sh` 相同：

```bash
python3 -m sglang.launch_server \
  --model-path /share_nfs/fanxingran/models/GLM-5.2-MXFP4 \
  --kv-cache-dtype fp8_e4m3 \
  --dsa-prefill-backend triton_gluon \
  --dsa-decode-backend triton_gluon \
  ...
```

工作树生效方式：

```bash
export PYTHONPATH=/path/to/sglang/python:${PYTHONPATH}
```

确认已经走到这份代码，在 server log 里应有：

```text
Using triton_gluon sparse MLA for DSA (prefill=triton_gluon, decode=triton_gluon, local heads=16).
```

`local heads` 在 TP4 是 16，TP8 是 8。对照 arm 使用 `triton`，日志里没有这一行。KV 跨度达到 2 GiB 时，dispatcher 还会打一次：

```text
triton_gluon: the KV cache exceeds 2 GiB per layer; kernels with 32-bit KV offsets are skipped
```

TP8/EP1/C1 的 pool 是 4,003,392 token（每层 2.31 GB），这行出现了，每张卡一条。TP4/EP4/C8 的 pool 是 2,484,224 token（每层 1.43 GB），这行是 0 条，全部走 gluon。

TP8/C1 不是整段退回 Triton：verify M=6 改用 `h8_m16`（`h8_m8` 有 `< 2**31` 字节的断言），draft M=1 仍走 `h8_m1`，只有 H8 prefill 整段退回 `triton_sparse_mla_fwd`。

## 单元测试

在容器里、单卡：

```bash
export PYTHONPATH=/path/to/sglang/python
export HIP_VISIBLE_DEVICES=0
python3 -m pytest -q \
  test/registered/kernels/ops/attention/test_gluon_sparse_mla.py \
  test/registered/unit/test_model_overrides.py -k "gluon or dsa"
```

2026-09-23 在上述镜像上：9 passed，58 subtests。覆盖 decode 行数、部分 `-1` slot、prefill 切段、约 4M 行的大 pool、FP8 query、CUDA graph replay，以及非 gfx950 / 非 fp8 / Triton < 3.8 时的报错。

## Kernel 微基准

方法：16 层不同的 query、KV、slots 抓进一张 HIP graph，回放 50 次取 p50。cold 是每层前 `zero_` 512 MB，把 L2 和 MALL 挤掉。top-k 2048 全有效，KV pool 每层 2^18 行。脚本在工作区 `glm52-mla-port/scripts/bench_sparse_mla_graph.py`，两次独立运行取平均。M=84 是同脚本补的一次运行。单位是每层微秒。

Verify 走 SGLang 的 `triton_sparse_mla_fwd`。Gluon 没有该行数的 kernel 时 pad 到下一个 2 的幂，空行 slot 填 `-1`。M = decode batch × 6（EAGLE 6 个 draft token）。

TP4（16 heads），cold：

| 并发 | M | Triton | Gluon | 加速 | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 20.5 | 13.5 | 1.53x | h16_m8 |
| 2 | 12 | 23.1 | 16.9 | 1.37x | h16_m16 |
| 4 | 24 | 29.9 | 22.0 | 1.36x | h16_m32 |
| 8 | 48 | 42.1 | 23.4 | 1.80x | h16_m64 |
| 10 | 60 | 43.1 | 24.8 | 1.74x | h16_m64 |
| 12 | 72 | 64.5 | 37.6 | 1.71x | h16_m128 |
| 14 | 84 | 64.7 | 37.4 | 1.73x | h16_m128 |

TP8（8 heads），cold：

| 并发 | M | Triton | Gluon | 加速 | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 19.2 | 12.1 | 1.58x | h8_m8 |
| 2 | 12 | 22.0 | 15.7 | 1.41x | h8_m16 |
| 4 | 24 | 28.8 | 19.6 | 1.47x | h8_m32 |
| 8 | 48 | 40.7 | 22.9 | 1.78x | h8_m64 |
| 10 | 60 | 41.5 | 26.3 | 1.58x | h8_m64 |
| 12 | 72 | 62.8 | 37.9 | 1.66x | h8_m128 |
| 14 | 84 | 61.9 | 38.9 | 1.59x | h8_m128 |

Draft decode 走 `triton_sparse_mla_decode_splitk`，每 cycle 5 次、每次 1 层。TP4 cold：M=1 是 13.2 对 8.7 µs（1.52x），M=2 是 14.5 对 10.2（1.42x），M=8 是 20.2 对 14.7（1.38x）。TP8 cold：M=1 是 12.7 对 8.0（1.58x）。

Prefill（生产形态的 top-k，页大小 64，TP4）加速大约 1.0–1.26 倍，overlap 很低时接近 1.0，均匀 slot 时有的点低于 1。脚本是 `glm52-mla-port/scripts/bench_prefill_real.py`。

相对 FP32 的 rel L2：Triton 约 2.7e-2（Q 和 P 量化成 FP8），Gluon 约 1.7e-3（TP4）到 2.6e-3（TP8）。

## 端到端 A/B

Harness 在 `glm52-mla-port/ab/`（不在本 git 仓库里）。Server 配置复制 InferenceX `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh`，只替换两个 `--dsa-*-backend`。两次 arm 除 backend 外相同。

公共项：GLM-5.2-MXFP4，`--kv-cache-dtype fp8_e4m3`，`--mem-fraction-static 0.85`，`--chunked-prefill-size 32768`，EAGLE `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`，`SGLANG_SIMULATE_ACC_LEN=3.61`。

| 格子 | 并行 | KV | 时长 / 题量 |
|---|---|---|---|
| GSM8K TP8/EP1 | TP8 EP1，`--max-running-requests 64` | 无 offload | 全部 1319 题 |
| GSM8K TP4/EP4 | TP4 EP4 | HiCache dram，180 GB，write_through | 全部 1319 题 |
| AgentX TP8/EP1/C1 | TP8 EP1，`--max-running-requests 2` | 无 offload | aiperf 1200 s |
| AgentX TP4/EP4/C8 | TP4 EP4，`--max-running-requests 16` | HiCache dram | aiperf 1200 s |

AgentX 客户端是官方场景，不是把 decode batch 钉死在 concurrency 上：

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --concurrency ${CONC} --benchmark-duration 1200 \
  --random-seed 42 --use-server-token-count \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --num-dataset-entries 393 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 --failed-request-threshold 0.1
```

`--concurrency` 是同时存活的会话树数量。trace 里有空档，所以 GPU 上的 decode batch 低于这个数。TP4/C8 实测 effective decode concurrency 平均 1.8（p50 是 1，p90 是 4，最大 8）。指标用 InferenceX 自己的 `python3 -m infx.results.agentic.process_agentic_result` 聚合，和网站同一套字段。

原始日志在 `glm52-mla-port/ab/results/<backend>-<phase>-.../`，聚合 JSON 在 `glm52-mla-port/ab/infx_agg/`。

### GSM8K（1319 题，flexible exact match）

| 配置 | triton | triton_gluon |
|---|---:|---:|
| TP8/EP1 | 0.927 | 0.938 |
| TP4/EP4 | 0.931 | 0.938 |

### AgentX TP4/EP4/C8（HiCache dram，1200 s）

| 指标 | triton | triton_gluon | 变化 |
|---|---:|---:|---:|
| P90 Interactivity (tok/s/user) | 95.42 | 96.88 | +1.5% |
| P50 Interactivity | 140.76 | 144.54 | +2.7% |
| Mean Interactivity | 129.18 | 133.62 | +3.4% |
| P90 E2E-norm Interactivity | 72.50 | 68.54 | −5.5% |
| TTFT p90 (s) | 1.067 | 1.127 | +5.6% |
| TTFT p50 (s) | 0.346 | 0.373 | +7.8% |
| TPOT mean / p50 / p90 / p95 (ms) | 7.74 / 7.10 / 10.48 / 12.71 | 7.48 / 6.92 / 10.32 / 11.93 | −3.4% / −2.5% / −1.5% / −6.1% |
| E2E latency p90 (s) | 11.01 | 10.32 | −6.3% |
| Throughput/GPU total (tok/s) | 10234 | 10114 | −1.2% |
| Output throughput (tok/s) | 249.8 | 240.1 | −3.9% |
| ISL / OSL 平均 | 108823 / 668 | 112149 / 669 | +3.1% / +0.2% |
| GPU / theoretical cache hit | 97.1% / 97.2% | 97.4% / 97.5% | 持平 |
| MTP accept len | 3.61 | 3.61 | 持平 |
| 成功请求 / error dropped | 456 / 0 | 432 / 0 |  |

### AgentX TP8/EP1/C1（无 offload，1200 s）

| 指标 | triton | triton_gluon | 变化 |
|---|---:|---:|---:|
| P90 Interactivity (tok/s/user) | 241.20 | 232.87 | −3.5% |
| P50 Interactivity | 263.53 | 255.89 | −2.9% |
| Mean Interactivity | 250.96 | 247.62 | −1.3% |
| P90 E2E-norm Interactivity | 61.32 | 66.85 | +9.0% |
| TTFT p90 (s) | 1.926 | 1.870 | −2.9% |
| TTFT p50 (s) | 0.541 | 0.599 | +10.7% |
| TPOT mean / p50 / p90 (ms) | 3.98 / 3.79 / 4.15 | 4.04 / 3.91 / 4.29 | +1.5% / +3.2% / +3.4% |
| E2E latency p90 (s) | 9.53 | 9.03 | −5.3% |
| Throughput/GPU total (tok/s) | 2613 | 2534 | −3.0% |
| Output throughput (tok/s) | 104.8 | 111.4 | +6.3% |
| ISL / OSL 平均 | 167216 / 843 | 166939 / 922 | −0.2% / +9.4% |
| GPU / theoretical cache hit | 95.6% / 95.8% | 95.5% / 95.8% | 持平 |
| MTP accept len | 3.61 | 3.61 | 持平 |
| 成功请求 / error dropped | 149 / 0 | 148 / 0 |  |

## 怎么读这些数

Kernel 的 1.4–1.8 倍是每层大约 20 µs 的 sparse MLA。Verify 是 78 层目标模型加 1 层 draft extend，各算一次：79 × 20.5 µs ≈ 1.62 ms。TP4/C8 实测一步是 ITL 7.74 ms × accept length 3.61 ≈ 27.9 ms。这 1.62 ms 只占约 6%，Gluon 省下约 0.55 ms，对应 ITL 大约 −2%。P90 Interactivity +1.5% 和这个上限一致。

官方 AgentX 的 C=8 没有把 decode batch 维持在 8，实际 batch 大约是 1–2，跑的是表里 M=6 和 M=12，不是 M=48 的 1.80 倍。

TP8/C1 的 1.58 倍来自 `h8_m8`。2.31 GB 的 pool 让它断言失败，serving 改走 `h8_m16`（microbench 上仍是 12.6 µs 对 Triton 的 19.2 µs）和 `h8_m1`。预期仍是小幅变快。榜单口径的 P90 Interactivity 是 −3.5%（232.87 对 241.20 tok/s/user），TPOT mean 是 +1.5%。方向和 kernel 预期相反，幅度小于这一臂 OSL +9.4% 的回放差异，每组只有一次 1200 s。

把 verify 的 sparse MLA 时间算成 0，这一档也只剩大约 6% 的 ITL。Decode kernel 再快，AgentX interactivity 的空间就在这里。Prefill 上 Gluon 没有稳定优势；同模型上改 Triton launch 的 [sglang#39059](https://github.com/sgl-project/sglang/pull/39059) 动的是 TTFT，和这条路不是一回事。
