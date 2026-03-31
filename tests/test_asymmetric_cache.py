import mlx.core as mx
from mlx_core.cache import TurboQuantKVCache

def test_asymmetric_cache_mixed_precision():
    head_dim = 128
    n_kv_heads = 8
    
    # 1. Инициализация (sink_size = 50)
    cache = TurboQuantKVCache(head_dim=head_dim, n_kv_heads=n_kv_heads, pq_bits=3, qjl_features=2048, fp16_sink_size=50)
    
    # 2. Создаем фейковый контекст (batch=1, heads=8, seq=150, dim=128)
    key1 = mx.random.key(42)
    fake_keys = mx.random.normal((1, n_kv_heads, 150, head_dim), key=key1)
    fake_values = mx.random.normal((1, n_kv_heads, 150, head_dim), key=key1)
    
    # 3. Эмуляция генерации (prefill 150 токенов разом)
    ret_keys, ret_values = cache.update_and_fetch(fake_keys, fake_values)
    
    # 4. Проверяем, что кэш чанкировался: 50 токенов в fp16 sink, осталось 100. 100 -> 1 чанк (64) + 36 в буфере.
    assert cache.sink_keys.shape[2] == 50
    assert len(cache.compressed_keys_chunks) == 1
    assert len(cache.compressed_values_chunks) == 1
    assert cache.key_buffer.shape[2] == 36
    
    # 5. Убеждаемся, что форма возвращаемых значений (sink + chunk + buf) в сумме 150
    assert ret_keys.shape == (1, n_kv_heads, 150, head_dim)
    assert ret_values.shape == (1, n_kv_heads, 150, head_dim)
    
    print("Mixed Precision Asymmetric KVCache tests passed successfully!")

def test_cache_lazy_init_no_args():
    """TurboQuantKVCache() without args should infer dims from first update_and_fetch call (Qwen compat)."""
    head_dim = 128
    n_kv_heads = 8

    cache = TurboQuantKVCache()

    assert cache.head_dim == 0
    assert cache.n_kv_heads == 0
    assert cache._initialized is False

    key1 = mx.random.key(42)
    fake_keys = mx.random.normal((1, n_kv_heads, 150, head_dim), key=key1)
    fake_values = mx.random.normal((1, n_kv_heads, 150, head_dim), key=key1)

    ret_keys, ret_values = cache.update_and_fetch(fake_keys, fake_values)

    assert cache._initialized is True
    assert cache.head_dim == head_dim
    assert cache.n_kv_heads == n_kv_heads
    assert cache.k_compressor is not None
    assert cache.v_compressor is not None

    assert ret_keys.shape == (1, n_kv_heads, 150, head_dim)
    assert ret_values.shape == (1, n_kv_heads, 150, head_dim)

    assert cache.sink_keys.shape[2] == 128
    assert cache.key_buffer.shape[2] == 22  # 150 - 128 sink = 22, < chunk_size(64)
    assert len(cache.compressed_keys_chunks) == 0

    print("Lazy init (no args) test passed!")


def test_cache_step_attribute():
    """step class attribute must be 256 for mlx_lm server compat."""
    assert TurboQuantKVCache.step == 256


def test_cache_keys_values_fallback():
    """keys/values fallback attrs must exist and be None initially."""
    cache = TurboQuantKVCache()
    assert cache.keys is None
    assert cache.values is None


def test_cache_incremental_tokens():
    """Simulate real LLM inference: prefill + token-by-token decode."""
    head_dim = 128
    n_kv_heads = 8

    cache = TurboQuantKVCache(head_dim=head_dim, n_kv_heads=n_kv_heads, fp16_sink_size=50)

    key1 = mx.random.key(99)
    prefill_keys = mx.random.normal((1, n_kv_heads, 130, head_dim), key=key1)
    prefill_values = mx.random.normal((1, n_kv_heads, 130, head_dim), key=key1)

    ret_k, ret_v = cache.update_and_fetch(prefill_keys, prefill_values)
    assert ret_k.shape == (1, n_kv_heads, 130, head_dim)
    assert cache.offset == 130
    assert cache.sink_keys.shape[2] == 50  # first 50 in sink
    assert len(cache.compressed_keys_chunks) == 1  # 80 remaining: 64 compressed + 16 buffer
    assert cache.key_buffer.shape[2] == 16

    for i in range(5):
        token_k = mx.random.normal((1, n_kv_heads, 1, head_dim), key=mx.random.key(i))
        token_v = mx.random.normal((1, n_kv_heads, 1, head_dim), key=mx.random.key(i))
        ret_k, ret_v = cache.update_and_fetch(token_k, token_v)

    assert cache.offset == 135
    assert ret_k.shape == (1, n_kv_heads, 135, head_dim)
    assert ret_v.shape == (1, n_kv_heads, 135, head_dim)
    assert cache.key_buffer.shape[2] == 21  # 16 + 5 tokens

    print("Incremental tokens test passed!")


if __name__ == "__main__":
    test_asymmetric_cache_mixed_precision()
    test_cache_lazy_init_no_args()
    test_cache_step_attribute()
    test_cache_keys_values_fallback()
    test_cache_incremental_tokens()
