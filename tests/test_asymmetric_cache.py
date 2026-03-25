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

if __name__ == "__main__":
    test_asymmetric_cache_mixed_precision()
