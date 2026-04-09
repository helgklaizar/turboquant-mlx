#include <pybind11/pybind11.h>
#include <mlx/mlx.h>

namespace py = pybind11;
using namespace mlx::core;

class CartesianToPolar : public mlx::core::Primitive {
public:
    CartesianToPolar(mlx::core::StreamOrDevice s) : mlx::core::Primitive(s) {}

    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        // CPU implementation fallback
        throw std::runtime_error("CPU eval not implemented. Use GPU.");
    }

    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        // Metal GPU dispatch logic goes here.
        // 1. Get metal device and command encoder.
        // 2. Load 'polarquant.metal' library.
        // 3. Set pipeline state & buffers.
        // 4. Dispatch threads equal to array size / 2.
    }

    void print(std::ostream& os) override {
        os << "CartesianToPolar";
    }
};

array cartesian_to_polar_fused(array x, mlx::core::StreamOrDevice s = {}) {
    // We register the custom primitive into the MLX computational graph
    return array({x.shape()[0], x.shape()[1] / 2}, x.dtype(), std::make_shared<CartesianToPolar>(s), {x})[0];
}

PYBIND11_MODULE(polarquant_ext, m) {
    m.doc() = "MLX C++ Extension for PolarQuant";
    m.def("cartesian_to_polar_fused", &cartesian_to_polar_fused, py::arg("x"), py::arg("stream") = py::none(), "Fused Metal implementation of Cartesian to Polar transformation");
}
