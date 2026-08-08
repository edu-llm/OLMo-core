import torch

import olmo_core.nn.flash_pd_native  # noqa: F401


def _inputs(device: str = "cpu"):
    state = 16
    time = 5
    destination = torch.arange(state, dtype=torch.int16, device=device).view(1, 1, state)
    routes = torch.zeros(1, 1, time, dtype=torch.int16, device=device)
    dictionary = torch.randn(1, 1, state, state, device=device)
    selector = torch.randn(1, time, 1, 1, device=device)
    recurrence = [torch.randn(1, 1, time, state, device=device) for _ in range(4)]
    beta = torch.rand(1, 1, time, device=device)
    gamma = torch.rand(1, 1, time, device=device)
    return (
        dictionary,
        selector,
        destination,
        destination,
        routes,
        *recurrence,
        beta,
        gamma,
    )


def test_mamba3_siso_torch_library_has_fake_meta_and_autograd_registration():
    operator = "flash_pd_native::mamba3_siso"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "Meta")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "Autograd")

    meta = _inputs("meta")
    real, imag = torch.ops.flash_pd_native.mamba3_siso(
        *meta,
        0.7,
        1.3,
        32,
        1,
    )
    assert real.device.type == imag.device.type == "meta"
    assert real.shape == imag.shape == (1, 1, 5, 16)


def test_mamba3_siso_opaque_op_compiles_fullgraph_without_graph_break():
    def recurrence(*inputs):
        return torch.ops.flash_pd_native.mamba3_siso(
            *inputs,
            0.7,
            1.3,
            32,
            1,
        )

    inputs = _inputs()
    compiled = torch.compile(recurrence, backend="eager", fullgraph=True)
    expected = recurrence(*inputs)
    actual = compiled(*inputs)

    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value)
