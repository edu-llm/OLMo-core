import builtins

import pytest

import olmo_core.nn.mamba3.mamba3_ssd_api as api


def _fail_mamba_import(monkeypatch, failure: BaseException) -> None:
    real_import = builtins.__import__

    def import_with_failure(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("mamba_ssm"):
            raise failure
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)


def test_has_mamba3_returns_false_only_when_the_package_is_absent(monkeypatch):
    missing = ModuleNotFoundError("No module named 'mamba_ssm'", name="mamba_ssm")
    _fail_mamba_import(monkeypatch, missing)

    assert api.has_mamba3() is False


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("libselective_scan_cuda.so: undefined symbol: _ZN2at4_ops"),
        RuntimeError("Triton API mismatch while importing mamba3_siso_combined"),
        ModuleNotFoundError("No module named 'causal_conv1d'", name="causal_conv1d"),
    ],
)
def test_has_mamba3_preserves_broken_binary_and_transitive_import_diagnostics(monkeypatch, failure):
    _fail_mamba_import(monkeypatch, failure)

    with pytest.raises(type(failure)) as caught:
        api.has_mamba3()

    assert caught.value is failure
