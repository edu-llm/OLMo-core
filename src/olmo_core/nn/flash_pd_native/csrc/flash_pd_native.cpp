#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> flash_pd_native_forward_cuda(
    torch::Tensor destination,
    torch::Tensor inverse_destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor bias_real,
    torch::Tensor bias_imag,
    int64_t chunk_size,
    int64_t mode);

std::vector<torch::Tensor> flash_pd_native_backward_cuda(
    torch::Tensor destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor output_real,
    torch::Tensor output_imag,
    torch::Tensor grad_output_real,
    torch::Tensor grad_output_imag);

std::vector<torch::Tensor> flash_pd_native_mamba3_forward_cuda(
    torch::Tensor destination,
    torch::Tensor inverse_destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor value_real,
    torch::Tensor value_imag,
    torch::Tensor beta,
    torch::Tensor gamma,
    int64_t chunk_size,
    int64_t mode);

std::vector<torch::Tensor> flash_pd_native_paper_backward_cuda(
    torch::Tensor dictionary_logits,
    torch::Tensor selector_logits,
    torch::Tensor destination,
    torch::Tensor routes,
    torch::Tensor diagonal_real,
    torch::Tensor diagonal_imag,
    torch::Tensor bias_real,
    torch::Tensor bias_imag,
    torch::Tensor output_real,
    torch::Tensor output_imag,
    torch::Tensor grad_output_real,
    torch::Tensor grad_output_imag,
    torch::Tensor value_real,
    torch::Tensor value_imag,
    torch::Tensor beta,
    torch::Tensor gamma,
    double dictionary_temperature,
    double router_temperature,
    int64_t chunk_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &flash_pd_native_forward_cuda, "Native Flash PD forward (CUDA)");
    module.def("backward", &flash_pd_native_backward_cuda, "Native Flash PD backward (CUDA)");
    module.def(
        "mamba3_forward",
        &flash_pd_native_mamba3_forward_cuda,
        "Native Flash PD Mamba-3 SISO forward (CUDA)");
    module.def(
        "paper_backward",
        &flash_pd_native_paper_backward_cuda,
        "Native Flash PD Appendix-C backward (CUDA)");
}
