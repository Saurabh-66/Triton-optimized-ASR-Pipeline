# GLM-ASR Triton Template Code Walkthrough

This document explains what each file implements, what it calls, and how the code flows end to end. It is intended to support a detailed review of the template model and its Triton kernels.

**Scope**
- Folder: glm_asr_triton_template
- Focus: model structure, kernel usage, call flow, and review points

**File Details**

**__init__.py**
- Implements: Package bootstrap, local path injection, and default backend flags.
- Calls into: layers module to set global flags on Linear, MLP, EncoderMLP.
- Called by: Any import of the package.
- Notes: Sets Linear.BACKEND to "cublas" (Torch matmul path) and disables fused MLP kernels by default.

**README.md**
- Implements: Assignment overview and TODO list for kernels across attention, layers, and rope.
- Calls into: None.
- Called by: Users reading instructions.
- Notes: Highlights which kernels are meant to be implemented and suggests testing commands.

**model.py**
- Implements: Core GLM-ASR model, config dataclass, audio encoder, text decoder, projector, and generation loop.
- Calls into: layers.RMSNorm, layers.LayerNorm, layers.Linear, layers.Embedding, layers.MLP, layers.gelu; rope.RotaryEmbedding, rope.apply_rotary_pos_emb; attention.scaled_dot_product_attention, attention.MultiHeadAttention; conv.Conv1d.
- Called by: weight_loader.load_model_from_hf and any user code constructing the model.
- Notes: Encodes audio via Conv1d layers, applies partial RoPE in audio encoder, full RoPE in decoder, supports KV cache and pre-allocated KV buffers, and provides generate() with top-k sampling.

**attention.py**
- Implements: Triton attention kernels (scores, softmax, output), fused attention kernel, and MultiHeadAttention wrapper with GQA support.
- Calls into: Triton kernels and Torch fallback for attention computation; uses next_power_of_two and MAX_ATTENTION_DIM for kernel gating.
- Called by: model.AudioEncoderLayer (scaled_dot_product_attention) and model.DecoderLayer (MultiHeadAttention).
- Notes: Uses fused kernel when CUDA and head_dim within limits; otherwise uses kernel trio or Torch fallback. Supports attention_mask and causal masking.

**conv.py**
- Implements: Conv1d via im2col + Triton matmul, Conv1dSubsampler helper, and a local GELU.
- Calls into: conv1d_matmul_kernel for CUDA path, Torch einsum for CPU path, and local gelu for Conv1dSubsampler.
- Called by: model.AudioEncoder uses Conv1d; Conv1dSubsampler is defined but not used by model.py.
- Notes: Uses padded tiles to match power-of-two block sizes. Falls back to Torch when shapes are large or non-CUDA.

**layers.py**
- Implements: Core neural layers and Triton kernels: RMSNorm, LayerNorm, GELU, SiLU, Linear (TF32), Linear+GELU, SwiGLU, Embedding, Softmax, and a fused RMSNorm+Linear kernel. Includes helper activations and fused MLPs.
- Calls into: Triton kernels when CUDA; Torch fallbacks for non-CUDA or unsupported shapes; optional gating and fusion paths in MLP.
- Called by: model.py for normalization, linear layers, embeddings, activations, and MLPs. __init__.py sets defaults for backend and fusion flags.
- Notes: Linear.BACKEND drives kernel selection; fused MLP and EncoderMLP are available but disabled by default in __init__.py.

**rope.py**
- Implements: RotaryEmbedding with cached cos/sin, Triton kernel for frequency computation, and apply_rotary_pos_emb helpers.
- Calls into: compute_freqs_kernel on CUDA; Torch compute on CPU.
- Called by: model.AudioEncoder for partial RoPE and model.DecoderLayer for full RoPE.
- Notes: Supports partial rotary factor and auto-refresh of cached positions when sequence length grows.

**weight_loader.py**
- Implements: Loading weights from a HuggingFace GLM-ASR model into the Triton template model; configuration adapter from HF config.
- Calls into: model.GlmAsrModel construction, helper load_* functions for conv, linear, norms, and embeddings.
- Called by: External user code via load_model_from_hf().
- Notes: Expects HF state dict key names matching GLM-ASR HF layout; loads weights in float32.

**Code Flow (End to End)**
1. User constructs model via GlmAsrModel(config) or load_model_from_hf().
2. Audio path: input_features -> AudioEncoder Conv1d stack -> RoPE cache -> AudioEncoderLayer stack -> LayerNorm.
3. Projector path: audio features -> MultiModalProjector pool + Linear + GELU + Linear.
4. Text path: input_ids -> Embedding -> DecoderLayer stack with RoPE + attention + MLP -> RMSNorm.
5. Output path: decoder hidden states -> lm_head Linear -> logits.
6. Generation path: encode_audio -> insert audio embeddings into token sequence -> decode with KV cache -> sampling loop.

**Kernel Selection Flow**
1. If CUDA and shape constraints are satisfied, Triton kernels are used.
2. If CUDA but constraints not satisfied, mixed paths may be used (some Triton, some Torch).
3. If CPU, Torch fallbacks are used for all kernels.

**Review Focus Points**
- Correctness of shape handling in attention and RoPE (especially partial RoPE and GQA expansion).
- Consistency of KV cache behavior in DecoderLayer and TextDecoder.
- Alignment of conv padding/stride math with HF reference (audio length computation in encode_audio).
- Kernel gating logic for power-of-two or tile size constraints.
- Weight loading key mapping correctness in weight_loader.py.

**Local Test Entrypoints**
- attention.py, conv.py, layers.py, rope.py each include a __main__ test block for quick sanity checks.
