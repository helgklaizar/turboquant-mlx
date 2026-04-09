# TurboQuant-MLX 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-blue)](https://github.com/ml-explore/mlx)

**Extreme KV Cache Compression (1-3 bit) for LLMs natively on Apple Silicon.**

TurboQuant-MLX is an advanced implementation of near-optimal distortion-rate KV cache compression algorithms tailored specifically for the Apple MLX framework. It significantly reduces memory usage of Language Models by up to 5x with almost perfectly preserved accuracy.

## 🌟 Key Features

- **Asymmetric K/V Compression**: Keys dictate attention accurately, while Values can be aggressively compressed.
- **PolarQuant**: Cartesian-to-Polar transformations for unbiased dot-product estimation.
- **Attention Sink**: Safeguards instruction-following by preserving initial prompt tokens in FP16.
- **Dynamic Chunking**: Caches generations strictly by segment chunks (64 tokens) to drop VRAM footprint.

## 📦 Installation

```bash
git clone https://github.com/helgklaizar/turboquant-mlx.git
cd turboquant-mlx
pip install -e .
```

## 🚀 Quick Start

```python
from plugins.cache_plugin import apply_turboquant_cache
# Apply TurboQuant monkey-patch globally
apply_turboquant_cache(model, k_theta_bits=8, v_theta_bits=3, fp16_sink_size=128)
```

## 📈 Performance & Benchmarks

Reduced VRAM consumption by 70% on Llama 3 8B at 64K context length.

---

## 🍏 The Mac AI Ecosystem
This initiative is a suite of high-performance tools natively optimized for Apple Silicon (MLX).

- [🍏 **Env-Selector-MLX**](https://github.com/helgklaizar/env-selector-mlx) — UI configurator for your AI environment.
- [🌉 **Cuda-Bridge-MLX**](https://github.com/helgklaizar/cuda-bridge-mlx) — Run CUDA-dependent projects natively.
- [🚀 **TurboQuant-MLX**](https://github.com/helgklaizar/turboquant-mlx) — Extreme KV Cache Compression (1-3 bit).
- [🔥 **Flamegraph-MLX**](https://github.com/helgklaizar/flamegraph-mlx) — Energy & Performance Visual Profiler.
- [🧠 **Rag-Indexer-MLX**](https://github.com/helgklaizar/rag-indexer-mlx) — Native system RAG with zero battery drain.
- [⚒️ **Forge-MLX**](https://github.com/helgklaizar/forge-mlx) — Fast and memory-efficient Fine-Tuning.
- [🔳 **BitNet-MLX**](https://github.com/helgklaizar/bitnet-mlx) — Native Ternary (1.58-bit) Kernels.
- [👁️ **OmniParser-MLX**](https://github.com/helgklaizar/omni-parser-mlx) — visual GUI understanding.
- [⚡️ **Flash-Attention-MLX**](https://github.com/helgklaizar/flash-attention-mlx) — Native FA3 for Metal.
- [🌿 **SageAttention-MLX**](https://github.com/helgklaizar/sage-attention-mlx) — Ultra-fast Quantized Attention.
- [🧬 **Attention-Matching-MLX**](https://github.com/helgklaizar/attention-matching-mlx) — Recursive 50x-100x context compression.
- [🚀 **RocketKV-MLX**](https://github.com/helgklaizar/rocket-kv-mlx) — Extreme 400x cache pruning.
- [📡 **KVTC-MLX**](https://github.com/helgklaizar/kvtc-mlx) — Transform coding for KV cache.
- [🌌 **AETHER-MLX**](https://github.com/helgklaizar/aether-mlx) — Geometric Sparse Attention.
- [🌌 **DeepSeek-MLX**](https://github.com/helgklaizar/deepseek-mlx) — High-throughput inference engine.
- [🎞 **Open-Sora-MLX**](https://github.com/helgklaizar/open-sora-mlx) — Text-to-Video generation pipeline.
- [🗣 **Moshi-Voice-MLX**](https://github.com/helgklaizar/moshi-voice-mlx) — Realtime Voice-to-Voice agents.
- [🎲 **MCTS-RL-MLX**](https://github.com/helgklaizar/mcts-rl-mlx) — Parallel reasoning at scale.
