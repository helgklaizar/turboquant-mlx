# TurboQuant+ Insights Issue Draft

**Title:** Greetings from TurboQuant-MLX & Some Insights to Share! (System Prompt vs Boundary Layers)

**Body:**

Hi TheTom & all contributors of `turboquant_plus`!

First of all, **huge thanks** for the incredible insights from your experiments! We run the Python/Apple MLX port of TurboQuant (`turboquant_mlx`), and your findings were exactly what we needed to take our architecture to the next level:
- We immediately **dropped QJL**. You're absolutely right: it was just unnecessarily increasing variance and slowing the graph down. Pure PolarQuant with optimal centroids provides much better PPL!
- We implemented **Asymmetric K/V Compression** and **Boundary V** directly into our MLX monkey-patch based on your papers. 

While adapting your research natively to Apple MLX (Python, lazy graph compilation without custom C++/Metal shaders), we noticed some interesting differences. We wanted to share our approach and hear your thoughts on these architectural choices:

1. **Attention Sink vs Boundary V:** While you protect the first 2 and last 2 **layers** (Boundary V), we *also* implemented an **Attention Sink (Heavy Hitter Caching)** where we strictly preserve the first ~128 **tokens** across *all layers* in uncompressed FP16. This is usually the System Prompt ("you are a helpful assistant and should..."), preventing instruction-following collapse at extreme contexts. Has `turboquant_plus` considered skipping compression for the *initial context tokens* across the board, or does Boundary V inherently solve the quality gap well enough that token-sinking is redundant?
2. **Dynamic Chunking Buffer (OOM prevention):** On Apple Silicon, we found that 70B+ models often face Out-Of-Memory limits during layer-by-layer KV streaming if the cache is fully pre-allocated at startup. We bypass this by buffering and compressing the KV cache dynamically in small 64-token chunks on the fly. Does `llama.cpp`'s native memory manager inherently avoid this by allocating static compact blocks, or are you utilizing dynamic chunk growth during the pre-fill phase as well?
3. **Sparse V Dequantization:** We loved the idea of gaining +23% decode speed by skipping V block dequantization for tokens with `attention_weight < 1e-6`. Unfortunately, since we operate strictly in `mlx.core` lazy Python graphs (without injecting custom `.metal` shaders), this dynamic sparsity mask is expensive to emulate inside standard dense matmuls. Huge win for the native C++ approach!

Thanks again for the incredible research, rigorous hardware validation, and the meticulous documentation. Great work from the community!
