import numpy as np

class QJLCompressor:
    def __init__(self, feature_dim: int, num_features: int, seed: int = 42):
        """
        Quantized Johnson-Lindenstrauss compressor (1-bit KV Cache).
        
        :param feature_dim: dimensionality of the source vectors (d)
        :param num_features: number of random features (k) in projection
        :param seed: seed for reproducible projection
        """
        self.feature_dim = feature_dim
        self.num_features = num_features
        np.random.seed(seed)
        
        # Random projection matrix
        # Elements sampled from N(0, 1)
        self.R = np.random.randn(feature_dim, num_features)
        
    def compress(self, x: np.ndarray):
        """
        Compresses a vector or batch of vectors into a 1-bit representation.
        In practice, 1-bit (sign) can be packed into int8 (8 values per byte).
        In the prototype we use float32/int8 arrays for clarity (values 1 and -1).
        
        :param x: 1D vector (d,) or batch of vectors (b, d)
        :return: tuple (x_quant, norm_x)
        """
        if x.ndim == 1:
            norm_x = np.linalg.norm(x)
            projected = np.dot(x, self.R)
        else:
            norm_x = np.linalg.norm(x, axis=1, keepdims=True)
            projected = np.dot(x, self.R)
            
        x_quant = np.sign(projected)
        # Handle edge-case where projection is exactly 0
        x_quant[x_quant == 0] = 1.0
        
        return x_quant, norm_x
        
    def estimate_dot(self, x_quant: np.ndarray, norm_x, y: np.ndarray) -> np.ndarray:
        """
        Asymmetric dot product estimation where one vector is quantized (x) and 
        the other is the attention query without quantization (y).
        
        :param x_quant: quantized feature vector with values {-1, 1}
        :param norm_x: L2 norm or vector of batch norms
        :param y: query vector (float)
        """
        y_proj = np.dot(y, self.R)
        
        # Direct matrix multiplication.
        # If x_quant (b, k) and y_proj (k,) -> yields (b,)
        # If both (k,) -> yields a scalar
        if x_quant.ndim == 2 and y_proj.ndim == 1:
            dot_product = np.dot(x_quant, y_proj)
        else:
            # For other shapes (e.g. (b, k) and (m, k) -> (b, m))
            dot_product = np.dot(x_quant, y_proj.T)
            
        scaling_factor = (norm_x / self.num_features) * np.sqrt(np.pi / 2)
        
        # Squeeze dimension (b, 1) for standard broadcasting with (b,)
        if dot_product.ndim == 1 and getattr(scaling_factor, 'ndim', 0) > 1:
            scaling_factor = np.squeeze(scaling_factor)
            
        return dot_product * scaling_factor
