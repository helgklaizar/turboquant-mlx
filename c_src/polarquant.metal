#include <metal_stdlib>
using namespace metal;

kernel void cartesian_to_polar_layer(
    device const float* in_data [[buffer(0)]],
    device float* radius_out [[buffer(1)]],
    device float* angle_out [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    float even = in_data[index * 2];
    float odd  = in_data[index * 2 + 1];
    
    radius_out[index] = sqrt(even * even + odd * odd);
    angle_out[index]  = atan2(odd, even);
}
