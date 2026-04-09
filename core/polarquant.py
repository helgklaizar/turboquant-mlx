import numpy as np

class PolarQuantCompressor:
    def __init__(self, feature_dim: int, bits: int = 3, seed: int = 42):
        """
        PolarQuant Compressor.
        Uses orthogonal random rotation (preconditioning) 
        and recursive transformation to polar coordinates with angle quantization.
        
        :param feature_dim: vector dimensionality (must be a power of two)
        :param bits: number of bits for angle quantization
        """
        self.feature_dim = feature_dim
        self.bits = bits
        self.max_idx = (1 << bits) - 1
        
        # Check that dim is a power of two
        assert (feature_dim & (feature_dim - 1)) == 0 and feature_dim > 0, "feature_dim must be a power of 2"
        
        np.random.seed(seed)
        # Generation of a random orthogonal matrix for Preconditioning
        # QR decomposition yields an orthogonal matrix
        H = np.random.randn(feature_dim, feature_dim)
        Q, R = np.linalg.qr(H)
        d = np.diagonal(R)
        self.R = Q * np.sign(d)
        
    def _quantize_angle(self, angle: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        # Linear scaling to [0, 1] -> to int [0, 2^b - 1]
        normalized = (angle - v_min) / (v_max - v_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        quantized = np.round(normalized * self.max_idx).astype(np.int8)
        return quantized
        
    def _dequantize_angle(self, q_angle: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        normalized = q_angle.astype(np.float32) / self.max_idx
        return normalized * (v_max - v_min) + v_min

    def _cartesian_to_polar_recursive(self, x: np.ndarray):
        """
        Recursive transformation of batch x into polar coordinates.
        x: (batch, dim)
        Returns a list of angles arrays and the final array of radii (batch, 1).
        """
        current = x
        angles_list = []
        layer = 0
        
        while current.shape[1] > 1:
            even = current[:, 0::2]
            odd = current[:, 1::2]
            
            radius = np.sqrt(even**2 + odd**2)
            angle = np.arctan2(odd, even)
            
            if layer == 0:
                # Original vectors lie within [-pi, pi]
                q_angle = self._quantize_angle(angle, -np.pi, np.pi)
            else:
                # Radii >= 0, thus the arctangent of two radii is in [0, pi/2]
                q_angle = self._quantize_angle(angle, 0.0, np.pi/2)
                
            angles_list.append(q_angle)
            current = radius
            layer += 1
            
        return angles_list, current

    def _polar_to_cartesian_recursive(self, angles_list: list, radius: np.ndarray):
        """
        Reconstruction. 
        Input: list of angles, radius (b, 1).
        """
        current = radius
        # Traverse from the last layer (root) to the zeroth (leaves)
        for layer in range(len(angles_list)-1, -1, -1):
            q_angle = angles_list[layer]
            
            # Dequantization
            if layer == 0:
                angle = self._dequantize_angle(q_angle, -np.pi, np.pi)
            else:
                angle = self._dequantize_angle(q_angle, 0.0, np.pi/2)
                
            even = current * np.cos(angle)
            odd = current * np.sin(angle)
            
            # Interleave elements: even, odd, even, odd
            b, dim = current.shape
            next_current = np.empty((b, dim * 2), dtype=np.float32)
            next_current[:, 0::2] = even
            next_current[:, 1::2] = odd
            current = next_current
            
        return current

    def compress(self, x: np.ndarray) -> dict:
        """
        Compression of a batch or a single vector.
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)
            
        # Rotation via orthogonal matrix (preconditioning)
        rotated = np.dot(x, self.R)
        
        # Extract quantized angles and root (radius)
        angles_list, radius = self._cartesian_to_polar_recursive(rotated)
        
        if is_single:
            angles_list = [a[0] for a in angles_list]
            radius = radius[0, 0]
            
        return {"angles": angles_list, "radius": radius}

    def decompress(self, compressed: dict) -> np.ndarray:
        """
        Decompression (approximate reconstruction of the original vector).
        """
        angles_list = compressed["angles"]
        radius = compressed["radius"]
        
        # Check if single vector or batch
        is_single = np.isscalar(radius) or (isinstance(radius, np.ndarray) and radius.ndim == 0)
        
        if is_single:
            radius_b = np.array([[radius]], dtype=np.float32)
            angles_b = [np.expand_dims(a, 0) for a in angles_list]
        else:
            radius_b = radius
            angles_b = angles_list
            
        # Inverse polar transformation
        rotated_approx = self._polar_to_cartesian_recursive(angles_b, radius_b)
        
        # Inverse rotation R^T (since the matrix is orthogonal, R^-1 = R^T)
        original_approx = np.dot(rotated_approx, self.R.T)
        
        if is_single:
            original_approx = original_approx[0]
            
        return original_approx
