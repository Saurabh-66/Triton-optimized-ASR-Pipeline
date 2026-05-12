# Triton-Optimized ASR Pipeline (GLM-ASR)

A GPU-accelerated automatic speech recognition pipeline built on top of
[GLM-ASR](https://huggingface.co/zai-org/glm-asr) (by zai-org / THUDM),
optimized using custom [Triton](https://github.com/openai/triton) kernels
to reduce inference latency by **3.7× (72.2% speedup)** over the baseline
PyTorch implementation — from **1480ms → 399ms** - while maintaining
**100% transcription accuracy** on LibriSpeech test-clean.

---

## Model

**GLM-ASR** is an end-to-end decoder-only ASR model from zai-org (HuggingFace:
`zai-org/glm-asr`). The pipeline consists of:

- **Mel Spectrogram** - 128-bin log-mel frontend (16 kHz audio)
- **Conv Subsampler** - 2× temporal downsampling
- **Audio Encoder** - 32-layer bidirectional transformer (hidden size 1280,
  20 attention heads, non-causal attention, LayerNorm, GELU, RoPE)
- **Multimodal Projector** - pooling (factor 4) + two linear layers
  (1280 → 5120 → 4096 → 2048 dims, GELU)
- **Text Decoder** - 28-layer autoregressive transformer (hidden size 2048,
  16Q/4KV heads GQA, RMSNorm, SiLU, causal attention, RoPE)

---

## What Was Optimized

All optimizations target the autoregressive **text decoder**, which accounts
for **94% of end-to-end runtime** at baseline.

### Custom Triton Kernels Implemented

| Kernel | Type | Optimization |
|---|---|---|
| `silu_kernel` | Element-wise | Coalesced 1D grid, block size 256 |
| `gelu_kernel` | Element-wise | Coalesced 1D grid, block size 256 |
| `rmsnorm_kernel` | Reduction | Single-pass intra-tile row reduction |
| `layernorm_kernel` | Reduction | Single-pass intra-tile row reduction |
| `softmax_kernel` | Reduction | Numerically stable online max-reduce |
| `linear_kernel_tf32` | Tiled GEMM | TF32 tensor cores, autotuned tiles (32×128 optimal for M=750; 16×16 for M=1 decode) |
| `attention_fused_kernel` | Fused attention | FlashAttention-2 style: online softmax, QKᵀ + V accumulation in SRAM, causal masking as compile-time constant, GQA without KV expansion |
| `rmsnorm_linear_fused` | Fused reduction+GEMM | Eliminates intermediate HBM round-trip between RMSNorm and linear projection |
| `swiglu_fused_kernel` | Fused MLP | Gate+up projection + SiLU + elementwise multiply in single kernel launch |
| `compute_freqs_kernel` | RoPE | Rotary position embedding frequency computation |

### Key Optimizations

**FlashAttention-2 Style Fusion**
Replaced three separate attention kernels (score computation → softmax → output)
with a single fused kernel that keeps intermediate attention scores in on-chip
SRAM rather than HBM. Eliminates the O(n²) memory footprint between kernel
launches. Causal masking applied as a compile-time constant removing runtime
branching. Native GQA support: 16 query heads attend over 4 KV heads without
data copying.

**KV Cache (largest single gain: +72.2%)**
Without caching, the decoder recomputes attention over the full sequence at
every generation step - O(n²) complexity. Implemented static KV buffer
pre-allocated at the projector stage; each decode step writes only the new
token's K/V and attends over cached history - O(n) per step. This single
change accounts for the majority of the end-to-end speedup.

**Fused RMSNorm + Linear**
In the unfused version, RMSNorm writes its output to HBM and the following
linear layer reads it back - a redundant memory round-trip per decoder layer.
The fused kernel keeps the normalized vector in registers and feeds it directly
into the tiled GEMM, saving one HBM read/write per layer × 28 layers × every
decode step.

**Fused SwiGLU MLP**
The baseline MLP runs four separate kernel launches (gate proj → up proj →
SiLU → elementwise multiply), each with Python dispatch overhead and HBM
round-trips. The fused kernel executes all four operations in a single launch,
reducing MLP latency from 19.86ms → 10.43ms (47% reduction in isolation).

**Tile Size Tuning with Autotune**
Benchmarked 13 tile configurations across 5 matrix shapes on the H200 MIG
slice. Key finding: 128×128 tiles have 2.5× higher arithmetic intensity than
32×128 but lose to 32×128 on 4 of 5 shapes because 128×128 exhausts the
register file (64KB accumulator causing L2 spill) and launches only 60 thread
blocks across 16 SMs. 32×128 launches 240 blocks with a 16KB accumulator and
no spill, achieving 1.35–2.06× speedups.

---

## Results

### End-to-End Latency

| Implementation | Latency (ms) | ms/token | Accuracy |
|---|---|---|---|
| Baseline (PyTorch) | 1480.9 | 113.92 | 100% |
| Triton baseline | 1437.4 | 110.57 | 100% |
| + FlashAttention-2 | 1220.0 | 93.85 | 100% |
| + FA2 + Autotune + TF32 | 1207.5 | 92.88 | 100% |
| + Native GQA | 1206.1 | 92.78 | 100% |
| + Fused RMSNorm+Linear | 1195.3 | 91.95 | 100% |
| **+ KV Cache (final)** | **399.4** | **30.72** | **100%** |

**3.7× speedup over baseline. 72.2% latency reduction.**

### Per-Operator Comparison (nsys profiling)

| Operation | Baseline kernel | Optimized kernel | Speedup |
|---|---|---|---|
| Large-M matmul (encoder) | cuBLAS ampere_sgemm 565µs | linear_kernel_tf32 302µs | 1.87× |
| Attention | flash_attention_fwd 57µs | attention_fused_kernel 49µs | 1.17× |
| DtoD memory traffic | 65 MB | 12 MB | 5.3× less |
| GPU kernel utilisation | 74.2% | 91.4% | +17pp |
| GPU memory stall | 25.8% | 8.6% | −17pp |

### Multi-Sample Evaluation (LibriSpeech test-clean, 10 samples)

- Mean latency: 588.4ms ± 163.2ms (27.96ms/token)
- Average WER: 2.6% | Average accuracy: 97.4%
- Higher latency vs. single benchmark reflects longer audio (up to 9.0s vs. 3.5s)

### Profiling Notes

Tested on **Edinburgh HPC** (NVIDIA H200 MIG 1g.18gb slice — 16 SMs,
~8.1 TFLOPS FP32 peak, ~580 GB/s HBM bandwidth, ridge point 14.0 FLOPs/byte).
All decode-time kernels operate at AI < 0.63 FLOPs/byte — ~20× below the
ridge point — confirming the pipeline is uniformly **memory-bound** during
autoregressive decoding. The primary optimization lever is therefore reducing
memory traffic and kernel launch overhead, not compute throughput.

---

## Roofline Analysis

At M=1 decode time, all kernels fall far below the roofline:

- SiLU: 0.63 FLOPs/byte
- RMSNorm: 0.42 FLOPs/byte  
- Attention (score + softmax + output): 0.33–0.50 FLOPs/byte
- Linear projection (M=1): 0.50 FLOPs/byte

The key bottleneck is not kernel arithmetic throughput — it is Python
dispatch overhead, host-device synchronization, and HBM round-trips
between kernel launches. Each decoder layer takes ~1.6ms kernel time,
yet the observed per-step latency at baseline was 432ms. Kernel execution
accounts for only ~10% of decode time; the remaining ~90% is Python
dispatch and memory allocation overhead — making kernel fusion and KV
caching the correct optimization strategies, not throughput maximization.

---

## Setup

```bash
source utils/setup-triton.sh
pip install -r requirements.txt
```

## Run

```bash
# Single sample benchmark
cd src
./benchmark.sh glm_asr_triton_optimized

# Detailed layer-by-layer breakdown
./benchmark_detailed.sh glm_asr_triton_optimized --runs 3

# Multi-sample evaluation
python benchmark_multi.py glm_asr_triton_optimized \
    --audio-dir ./test_samples/audio \
    --n-samples 10 \
    --runs 5
```

## Directory Structure

```
├── src/
│   ├── glm_asr_triton_optimized/  # most optimized version (use this)
│   ├── glm_asr_triton_baseline/   # baseline Triton reference
│   ├── glm_asr_scratch/           # PyTorch CPU reference
│   ├── benchmark.sh               # single-sample benchmark entry point
│   ├── benchmark_multi.py         # multi-sample benchmark
│   ├── benchmark_student.py       # benchmark runner used by benchmark.sh
│   ├── test_audio.wav             # single-sample test audio
│   └── test_samples/
│       ├── audio/                 # 20 LibriSpeech sample clips
│       └── transcripts.txt        # expected transcripts
├── utils/
│   └── setup-triton.sh            # Triton environment setup
└── requirements.txt               # Python dependencies
```

---

## References

- Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS 2022.
- Shazeer. Fast Transformer Decoding: One Write-Head is All You Need. arXiv 2019.
- Williams et al. Roofline: An Insightful Visual Performance Model. CACM 2009.
- NVIDIA. TensorFloat-32 in the NVIDIA Ampere Architecture. 2020.
- Panayotov et al. LibriSpeech: An ASR Corpus. ICASSP 2015.
- GLM-ASR model: https://huggingface.co/zai-org/glm-asr





