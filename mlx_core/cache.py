import mlx.core as mx
from .mlx_turboquant import MLXTurboQuant

class TurboQuantKVCache:
    """
    KVCache реализация для Apple MLX.
    Заменяет стандартный mlx_lm.models.cache.KVCache на нашу сжатую версию TurboQuant.
    Она сжимает ключи (и значения по желанию) во время префил-фазы генерации.
    """
    def __init__(self, head_dim: int, n_kv_heads: int, pq_bits: int = 3, qjl_features: int = 2048):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        
        self.compressor = MLXTurboQuant(feature_dim=head_dim, pq_bits=pq_bits, qjl_features=qjl_features)
        self.offset = 0
        self.chunk_size = 64
        
        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []
        
        self.key_buffer = None
        self.value_buffer = None

    def _compress_and_store(self, k: mx.array, v: mx.array):
        # k shape: (batch_size, n_kv_heads, seq_len, head_dim)
        b, h, s, d = k.shape
        k_2d = mx.reshape(k, (-1, d))
        compressed_k = self.compressor.compress(k_2d)
        self.compressed_keys_chunks.append((compressed_k, (b, h, s, d)))
        
        v_2d = mx.reshape(v, (-1, d))
        compressed_v = self.compressor.compress(v_2d)
        self.compressed_values_chunks.append((compressed_v, (b, h, s, d)))

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.offset += keys.shape[2]
        
        if self.key_buffer is None:
            self.key_buffer = keys
            self.value_buffer = values
        else:
            self.key_buffer = mx.concatenate([self.key_buffer, keys], axis=2)
            self.value_buffer = mx.concatenate([self.value_buffer, values], axis=2)
            
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
                
        # На лету декомпрессируем старые чанки
        full_keys = []
        full_values = []
        
        for comp_k, shape in self.compressed_keys_chunks:
            k_approx_2d = self.compressor.decompress(comp_k)
            full_keys.append(mx.reshape(k_approx_2d, shape))
            
        for comp_v, shape in self.compressed_values_chunks:
            v_approx_2d = self.compressor.decompress(comp_v)
            full_values.append(mx.reshape(v_approx_2d, shape))
            
        if self.key_buffer is not None:
            full_keys.append(self.key_buffer)
            full_values.append(self.value_buffer)
            
        if not full_keys:
            return keys, values
            
        return mx.concatenate(full_keys, axis=2), mx.concatenate(full_values, axis=2)

def apply_turboquant_cache(model, bits: int = 3, qjl_features: int = 2048):
    """
    Monkey-patch / Hook для интеграции TurboQuant напрямик в любую LLM (Llama, Gemma) на mlx-lm.
    Пробегает по слоям модели и подменяет экземпляр cache.
    """
    count = 0
    # mlx.nn.Module имеет метод named_modules() для прохода по графу сети
    if not hasattr(model, 'named_modules'):
        print("[TurboQuant] Ошибка: Модель не является mlx.nn.Module")
        return
        
    for name, module in model.named_modules():
        # У слоев Attention (напр. LlamaAttention) есть атрибут cache
        if hasattr(module, "cache") and hasattr(module, "head_dim"):
            n_kv = getattr(module, "n_kv_heads", module.n_heads) # Фолбэк если не GQA
            
            module.cache = TurboQuantKVCache(
                head_dim=module.head_dim,
                n_kv_heads=n_kv,
                pq_bits=bits,
                qjl_features=qjl_features
            )
            count += 1
            
    print(f"[TurboQuant] KV-Кэш успешно подменён: {count} Attention-слоев переведено на {bits}-битное сжатие.")
