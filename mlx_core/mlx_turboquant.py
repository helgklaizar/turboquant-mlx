import mlx.core as mx
from mlx_core.mlx_qjl import MLXQJLCompressor
from mlx_core.mlx_polarquant import MLXPolarQuantCompressor

class MLXTurboQuant:
    def __init__(self, feature_dim: int, pq_bits: int = 3, qjl_features: int = 2048, seed: int = 42):
        self.feature_dim = feature_dim
        self.pq = MLXPolarQuantCompressor(feature_dim=feature_dim, bits=pq_bits, seed=seed)
        self.qjl = MLXQJLCompressor(feature_dim=feature_dim, num_features=qjl_features, seed=seed+1)
        
    def compress(self, x: mx.array) -> dict:
        pq_compressed = self.pq.compress(x)
        x_mse_approx = self.pq.decompress(pq_compressed)
        
        residual = x - x_mse_approx
        qjl_quant, qjl_norm = self.qjl.compress(residual)
        
        return {
            "pq_data": pq_compressed,
            "qjl_data": qjl_quant,
            "qjl_norm": qjl_norm
        }
        
    def estimate_dot(self, compressed: dict, y: mx.array) -> mx.array:
        x_mse_approx = self.pq.decompress(compressed["pq_data"])
        
        if x_mse_approx.ndim == 2 and y.ndim == 1:
            dot_mse = mx.matmul(x_mse_approx, y)
        elif x_mse_approx.ndim == 1 and y.ndim == 2:
            dot_mse = mx.matmul(y, x_mse_approx)
        elif x_mse_approx.ndim == 1 and y.ndim == 1:
            dot_mse = mx.matmul(x_mse_approx, y)
        else:
            dot_mse = mx.matmul(x_mse_approx, y.astype(x_mse_approx.dtype).T)
            
        dot_residual = self.qjl.estimate_dot(
            x_quant=compressed["qjl_data"], 
            norm_x=compressed["qjl_norm"], 
            y=y
        )
        
        return dot_mse + dot_residual
        
    def decompress(self, compressed: dict) -> mx.array:
        """
        Восстанавливает векторы (используя базовый MSE-квантователь).
        Примечание: Для абсолютно математически точного dot-product (учитывая QJL остаток) 
        следует использовать estimate_dot(), так как QJL остаток оценивается асимметрично.
        Здесь возвращаем PolarQuant базис.
        """
        return self.pq.decompress(compressed["pq_data"])
