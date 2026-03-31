import mlx.core as mx
from .mlx_turboquant import MLXTurboQuant


class TurboQuantKVCache:
    """
    KVCache implementation for Apple MLX with TurboQuant compression.
    Compresses keys via TurboQuant (PolarQuant + QJL) and values via PolarQuant.
    """
    step = 256  # match mlx_lm KVCache interface

    def __init__(self, head_dim: int = 0, n_kv_heads: int = 0, pq_bits: int = 3, qjl_features: int = 2048, fp16_sink_size: int = 128):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.pq_bits = pq_bits
        self.qjl_features = qjl_features

        self.k_compressor = None
        self.v_compressor = None

        self.offset = 0
        self.chunk_size = 64
        self.fp16_sink_size = fp16_sink_size

        self.sink_keys = None
        self.sink_values = None

        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []

        self.key_buffer = None
        self.value_buffer = None

        # Standard KVCache fallback fields for compatibility
        self.keys = None
        self.values = None

        self._initialized = head_dim > 0

    def _lazy_init(self, keys: mx.array):
        """Initialize compressors on first update_and_fetch call using actual tensor dimensions."""
        if self._initialized:
            return
        self.head_dim = keys.shape[-1]
        self.n_kv_heads = keys.shape[1]
        self.k_compressor = MLXTurboQuant(feature_dim=self.head_dim, pq_bits=self.pq_bits, qjl_features=self.qjl_features)
        from .mlx_polarquant import MLXPolarQuantCompressor
        self.v_compressor = MLXPolarQuantCompressor(feature_dim=self.head_dim, bits=self.pq_bits)
        self._initialized = True

    def _ensure_compressors(self):
        if self.k_compressor is None and self.head_dim > 0:
            self.k_compressor = MLXTurboQuant(feature_dim=self.head_dim, pq_bits=self.pq_bits, qjl_features=self.qjl_features)
            from .mlx_polarquant import MLXPolarQuantCompressor
            self.v_compressor = MLXPolarQuantCompressor(feature_dim=self.head_dim, bits=self.pq_bits)

    def _compress_and_store(self, k: mx.array, v: mx.array):
        self._ensure_compressors()
        b, h, s, d = k.shape
        k_2d = mx.reshape(k, (-1, d))
        compressed_k = self.k_compressor.compress(k_2d)
        self.compressed_keys_chunks.append((compressed_k, (b, h, s, d)))

        v_2d = mx.reshape(v, (-1, d))
        compressed_v = self.v_compressor.compress(v_2d)
        self.compressed_values_chunks.append((compressed_v, (b, h, s, d)))

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self._lazy_init(keys)

        prev_offset = self.offset
        self.offset += keys.shape[2]

        # 1. Attention Sink — keep first fp16_sink_size tokens uncompressed
        if prev_offset < self.fp16_sink_size:
            remaining_sink = self.fp16_sink_size - prev_offset

            k_sink_part = keys[:, :, :remaining_sink, :]
            v_sink_part = values[:, :, :remaining_sink, :]

            if self.sink_keys is None:
                self.sink_keys = k_sink_part
                self.sink_values = v_sink_part
            else:
                self.sink_keys = mx.concatenate([self.sink_keys, k_sink_part], axis=2)
                self.sink_values = mx.concatenate([self.sink_values, v_sink_part], axis=2)

            k_compress_part = keys[:, :, remaining_sink:, :]
            v_compress_part = values[:, :, remaining_sink:, :]
        else:
            k_compress_part = keys
            v_compress_part = values

        # 2. Compression for remaining tokens
        if k_compress_part.shape[2] > 0:
            if self.key_buffer is None:
                self.key_buffer = k_compress_part
                self.value_buffer = v_compress_part
            else:
                self.key_buffer = mx.concatenate([self.key_buffer, k_compress_part], axis=2)
                self.value_buffer = mx.concatenate([self.value_buffer, v_compress_part], axis=2)

            while self.key_buffer is not None and self.key_buffer.shape[2] >= self.chunk_size:
                chunk_k = self.key_buffer[:, :, :self.chunk_size, :]
                chunk_v = self.value_buffer[:, :, :self.chunk_size, :]

                self._compress_and_store(chunk_k, chunk_v)

                if self.key_buffer.shape[2] > self.chunk_size:
                    self.key_buffer = self.key_buffer[:, :, self.chunk_size:, :]
                    self.value_buffer = self.value_buffer[:, :, self.chunk_size:, :]
                else:
                    self.key_buffer = None
                    self.value_buffer = None

        # 3. Decompress old chunks into full cache for attention
        full_keys = []
        full_values = []

        if self.sink_keys is not None:
            full_keys.append(self.sink_keys)
            full_values.append(self.sink_values)

        for comp_k, shape in self.compressed_keys_chunks:
            k_approx_2d = self.k_compressor.decompress(comp_k)
            full_keys.append(mx.reshape(k_approx_2d, shape))

        for comp_v, shape in self.compressed_values_chunks:
            v_approx_2d = self.v_compressor.decompress(comp_v)
            full_values.append(mx.reshape(v_approx_2d, shape))

        if self.key_buffer is not None:
            full_keys.append(self.key_buffer)
            full_values.append(self.value_buffer)

        if not full_keys:
            return keys, values

        return mx.concatenate(full_keys, axis=2), mx.concatenate(full_values, axis=2)

    @property
    def state(self):
        k = []
        v = []
        if self.sink_keys is not None:
            k.append(self.sink_keys)
            v.append(self.sink_values)
        if self.key_buffer is not None:
            k.append(self.key_buffer)
            v.append(self.value_buffer)

        ret_k = mx.concatenate(k, axis=2) if k else mx.array([])
        ret_v = mx.concatenate(v, axis=2) if v else mx.array([])
        return ret_k, ret_v

    @property
    def memory_size(self):
        total_bytes = 0
        for t in [self.sink_keys, self.sink_values, self.key_buffer, self.value_buffer]:
            if t is not None:
                total_bytes += t.size * 2

        for comp, _ in self.compressed_keys_chunks:
            total_bytes += comp["pq_data"]["r_quant"].size * comp["pq_data"]["r_quant"].dtype.size
            total_bytes += comp["pq_data"]["theta_quant"].size * comp["pq_data"]["theta_quant"].dtype.size
            if "qjl_data" in comp:
                total_bytes += comp["qjl_data"].size * 1
                total_bytes += comp["qjl_norm"].size * 2

        for comp, _ in self.compressed_values_chunks:
            total_bytes += comp["pq_data"]["r_quant"].size * comp["pq_data"]["r_quant"].dtype.size
            total_bytes += comp["pq_data"]["theta_quant"].size * comp["pq_data"]["theta_quant"].dtype.size

        return total_bytes


def apply_turboquant_cache(model=None, bits: int = 3, qjl_features: int = 2048, fp16_sink_size: int = 128):
    """
    Monkey-patch to integrate TurboQuant into any mlx-lm model.
    Replaces KVCache in the model factory so all layers use compressed caches.
    """
    try:
        import mlx_lm.models.cache as cache_module
    except ImportError:
        print("[TurboQuant] Error: mlx-lm not installed.")
        return

    class PatchedCache(TurboQuantKVCache):
        def __init__(self, head_dim: int = 0, n_kv_heads: int = 0, **kwargs):
            super().__init__(
                head_dim=head_dim,
                n_kv_heads=n_kv_heads,
                pq_bits=bits,
                qjl_features=qjl_features,
                fp16_sink_size=fp16_sink_size
            )

    cache_module.KVCache = PatchedCache

    if hasattr(cache_module, 'make_prompt_cache'):
        _original_make = cache_module.make_prompt_cache

        def patched_make_prompt_cache(model, max_kv_size=None):
            if hasattr(model, "make_cache"):
                return model.make_cache()
            return [PatchedCache(head_dim=getattr(l, 'head_dim', 0), n_kv_heads=getattr(l, 'n_kv_heads', getattr(l, 'n_heads', 0))) for l in model.layers]

        cache_module.make_prompt_cache = patched_make_prompt_cache

    print(f"[TurboQuant] Patch applied: KVCache replaced with TurboQuant compression.")
    print(f"[TurboQuant] Settings: {bits}-bit cache, Attention Sink (FP16): first {fp16_sink_size} tokens.")
