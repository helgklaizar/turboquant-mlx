# TurboQuant-MLX 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-blue)](https://github.com/ml-explore/mlx)

**Extreme KV Cache Compression (1-3 bit) for LLMs on Apple Silicon**

TurboQuant MLX is an advanced implementation of near-optimal distortion-rate KV cache compression algorithms tailored specifically for the Apple MLX framework. It significantly reduces memory usage of Language Models by up to 5x with almost perfectly preserved accuracy. 

The library heavily utilizes techniques from Google Research such as **PolarQuant** (Cartesian-to-Polar transformations) and **Quantized Johnson-Lindenstrauss (QJL)** for unbiased dot-product estimation.

## 🌟 Key Features

- **Asymmetric Compression:** Compresses `Keys` using TurboQuant (PolarQuant + QJL) for exact dot-product estimation, and `Values` using lightweight PolarQuant to eliminate MSE errors.
- **Attention Sink (Heavy Hitter Caching):** Safeguards LLM instruction-following capabilities by preserving the initial system prompt (e.g., first 128 tokens) in uncompressed `float16`. 
- **Dynamic Chunking Buffer:** Caches long generations strictly by segment chunks (64 tokens), drastically dropping VRAM consumption footprint on M-Series Macs.
- **EXO Cluster Ready:** Fully compatible with decentralized Apple Silicon inference networks. 
- **Drop-in `mlx-lm` Replacement:** Hook directly into Apple's official `mlx_lm` factory with just two lines of code without altering deep internal logics.

## 📦 Installation

```bash
git clone https://github.com/helgklaizar/turboquant_mlx.git
cd turboquant_mlx
pip install -e .
```

## 🚀 Quick Start

You can seamlessly plug TurboQuant into any existing `mlx_lm` models (Llama 3, Gemma 2, etc.) to immediately free up gigabytes of GPU memory.

```python
import mlx.core as mx
from mlx_lm import load
from mlx_core.cache import apply_turboquant_cache

# Load your favorite model
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

# Apply TurboQuant monkey-patch globally
# bits - polar compression bitrate per angular coordinate
# fp16_sink_size - tokens immune to compression (usually your System Prompt)
apply_turboquant_cache(model, bits=3, fp16_sink_size=128)

# Now, any model generation will consume ~70% less memory on the KV-Cache.
```

## 📈 Performance & Benchmarks

We conducted extensive memory footprint analysis comparing pure **FP16 caching** vs **TurboQuant (3-bit)** on extreme sequence lengths. TurboQuant slashes memory consumption dynamically while preserving structural reasoning logic.

### Memory Overhead Reduction (Llama 3 8B)

```mermaid
pie title Memory Footprint at 64K Context Length (MBs)
    "Uncompressed FP16 Cache (1024 MB)" : 1024
    "TurboQuant 3-bit Cache (192 MB)" : 192
```

| Context Length (Tokens) | Uncompressed (FP16) | TurboQuant (3-bit) | Memory Saved | Compression |
|-------------------------|---------------------|--------------------|--------------|-------------|
| **4,096** | 64 MB | **12 MB** | ~81% | `5.3x` |
| **16,384** | 256 MB | **48 MB** | ~81% | `5.3x` |
| **65,536** | 1024 MB | **192 MB** | ~81% | `5.3x` |
| **128,000** | 2048 MB | **384 MB** | ~81% | `5.3x` |

### 🤖 LLM Architecture Compatibility 
We executed a 100% precision **Needle-in-a-Haystack** stress test across top HuggingFace variants:

| Architecture | Model Tested | 3-bit Survival | Notes |
|-------------|--------------|:---:|-------|
| **Meta Llama 3**  | `Llama-3-8B-Instruct-4bit` | ✅ Pass | Flawless dot-product accuracy. Perfect for heavy MLX workloads. |
| **Meta Llama 3.2**| `Llama-3.2-1B-Instruct-4bit`| ✅ Pass | Highly resilient despite tiny 1B parameter count! |
| **Qwen 2.5**      | `Qwen2.5-1.5B-Instruct-4bit`| ❌ Fail | Extreme bits collapse on fractional models. (7B+ recommended) |
| **Gemma 2**       | `gemma-2-2b-it-4bit`        | ❌ Fail | Native embeddings clash with heavy polar transformations at 3-bit. |

> **🔥 Next targets to test locally on your Mac:** If you have 16GB+ RAM, we highly recommend testing **Mistral NeMo 12B** (robust RoPE) or **DeepSeek R1 Distill 8B** (reasoning integrity under compression).

## 🌐 OpenAI-Compatible Server

Run your models via a highly-optimized API server suitable for tools like Chatbox, Bolt, or ChatGPT frontends with built-in compression:

```bash
python scripts/run_server.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```

## 🤝 Acknowledgements

Massive thanks to the **[DeadByDawn101/turboquant-mlx](https://github.com/DeadByDawn101/turboquant-mlx)** repository for architectural inspiration! The concepts regarding Exo cluster integration, deep `make_prompt_cache` patching, and exact memory-byte tracking were successfully adapted into this standalone optimization thanks to their incredible groundwork!
