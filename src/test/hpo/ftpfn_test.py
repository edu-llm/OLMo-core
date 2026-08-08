import sys
import types

import numpy as np
import pytest
import torch

import olmo_core.hpo.ftpfn as ftpfn_module
from olmo_core.hpo.ftpfn import (
    ObservedCurve,
    PosteriorInput,
    QueryPoint,
    TrustedFTPFN,
    assemble_posterior_input,
)
from olmo_core.hpo.objective import CENormalizer
from olmo_core.hpo.types import CurvePoint


def _norm():
    return CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)


def test_assemble_normalizes_time_and_objective_with_stable_ids():
    target = 4096
    observed = [
        ObservedCurve(
            curve_id=1,
            unit_config=(0.2, 0.8),
            points=(CurvePoint(1024, 5.0), CurvePoint(2048, 4.0)),
        ),
        ObservedCurve(curve_id=2, unit_config=(0.6, 0.1), points=(CurvePoint(1024, 4.5),)),
    ]
    queries = [
        QueryPoint(curve_id=1, unit_config=(0.2, 0.8), t=3072 / target),  # continuation
        QueryPoint(curve_id=0, unit_config=(0.9, 0.9), t=1024 / target),  # brand-new config
    ]
    x = assemble_posterior_input(observed, queries, target_tokens=target, normalizer=_norm())
    assert isinstance(x, PosteriorInput)
    # t = tokens / target, in (0, 1].
    assert x.context_t[0] == pytest.approx(1024 / target)
    assert x.context_t[-1] == pytest.approx(1024 / target)
    # y is the FT-PFN objective (higher is better), ce=4.0 -> 0.5.
    assert x.context_y[1] == pytest.approx(0.5)
    # context ids preserved; query for a new config keeps id 0.
    assert set(x.context_ids.tolist()) == {1, 2}
    assert x.query_ids.tolist() == [1, 0]
    assert x.context_hp.shape == (3, 2)
    assert x.query_hp.shape == (2, 2)


def test_assemble_rejects_non_finite_ce():
    with pytest.raises(ValueError):
        assemble_posterior_input(
            [ObservedCurve(1, (0.5,), (CurvePoint(1024, float("nan")),))],
            [QueryPoint(1, (0.5,), 0.5)],
            target_tokens=2048,
            normalizer=_norm(),
        )


def test_assemble_rejects_time_outside_unit_interval():
    with pytest.raises(ValueError):
        assemble_posterior_input(
            [ObservedCurve(1, (0.5,), (CurvePoint(4096, 3.0),))],
            [QueryPoint(1, (0.5,), 0.5)],
            target_tokens=2048,  # 4096/2048 = 2.0 > 1
            normalizer=_norm(),
        )


def test_assemble_rejects_more_than_ten_hp_dims():
    wide = tuple([0.5] * 11)
    with pytest.raises(ValueError):
        assemble_posterior_input(
            [ObservedCurve(1, wide, (CurvePoint(1024, 3.0),))],
            [QueryPoint(1, wide, 0.5)],
            target_tokens=4096,
            normalizer=_norm(),
        )


def test_assemble_rejects_too_many_context_curves():
    observed = [ObservedCurve(i + 1, (i / 2000.0,), (CurvePoint(1024, 3.0),)) for i in range(1001)]
    with pytest.raises(ValueError):
        assemble_posterior_input(observed, [], target_tokens=4096, normalizer=_norm())


def test_assemble_rejects_hp_outside_unit_cube():
    with pytest.raises(ValueError):
        assemble_posterior_input(
            [ObservedCurve(1, (1.5,), (CurvePoint(1024, 3.0),))],
            [],
            target_tokens=4096,
            normalizer=_norm(),
        )


def test_assemble_rejects_curve_id_reused_for_different_config():
    observed = [
        ObservedCurve(1, (0.1,), (CurvePoint(1024, 4.0),)),
        ObservedCurve(1, (0.9,), (CurvePoint(2048, 3.5),)),
    ]
    with pytest.raises(ValueError):
        assemble_posterior_input(observed, [], target_tokens=4096, normalizer=_norm())


def test_assemble_enforces_reserved_and_matching_curve_ids():
    with pytest.raises(ValueError):
        assemble_posterior_input(
            [ObservedCurve(0, (0.1,), (CurvePoint(1024, 4.0),))],
            [],
            target_tokens=4096,
            normalizer=_norm(),
        )
    observed = [ObservedCurve(1, (0.1,), (CurvePoint(1024, 4.0),))]
    for query in (QueryPoint(2, (0.2,), 0.5), QueryPoint(1, (0.2,), 0.5)):
        with pytest.raises(ValueError):
            assemble_posterior_input(observed, [query], target_tokens=4096, normalizer=_norm())


def test_assemble_rejects_config_identity_ambiguity_in_real_ifbo_tokenizer():
    duplicated_config = [
        ObservedCurve(1, (0.1,), (CurvePoint(1024, 4.0),)),
        ObservedCurve(2, (0.1,), (CurvePoint(2048, 3.5),)),
    ]
    with pytest.raises(ValueError):
        assemble_posterior_input(duplicated_config, [], target_tokens=4096, normalizer=_norm())

    observed = [ObservedCurve(1, (0.1,), (CurvePoint(1024, 4.0),))]
    with pytest.raises(ValueError):
        assemble_posterior_input(
            observed,
            [QueryPoint(0, (0.1,), 0.5)],
            target_tokens=4096,
            normalizer=_norm(),
        )


def test_trusted_ftpfn_requires_ifbo():
    ifbo = pytest.importorskip("ifbo")  # noqa: F841 - skip cleanly when optional dep absent
    from olmo_core.hpo.ftpfn import TrustedFTPFN

    # We do not download weights in the unit test; constructing with a bogus dir and
    # verify=True must fail closed rather than silently loading an unverified artifact.
    with pytest.raises(Exception):
        TrustedFTPFN(artifact_path="/nonexistent/ftpfn.pt", verify=True)


def test_trusted_ftpfn_loads_exact_verified_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / "verified.pt"
    artifact.write_bytes(b"verified")
    calls = {}

    class FakeFTPFN(torch.nn.Module):
        pass

    class FakeInner:
        def eval(self):
            calls["eval"] = True

    fake_ifbo = types.ModuleType("ifbo")
    fake_surrogate = types.ModuleType("ifbo.surrogate")
    fake_surrogate.FTPFN = FakeFTPFN
    monkeypatch.setitem(sys.modules, "ifbo", fake_ifbo)
    monkeypatch.setitem(sys.modules, "ifbo.surrogate", fake_surrogate)
    monkeypatch.setattr(TrustedFTPFN, "_verify_checksum", staticmethod(lambda path: None))

    def fake_load(path, **kwargs):
        calls["path"] = str(path)
        calls["kwargs"] = kwargs
        return FakeInner()

    monkeypatch.setattr(torch, "load", fake_load)
    model = TrustedFTPFN(str(artifact), device="cpu")

    assert calls["path"] == str(artifact)
    assert calls["kwargs"]["map_location"] == torch.device("cpu")
    assert calls["kwargs"]["weights_only"] is False
    assert calls["eval"] is True
    assert model._model.device == torch.device("cpu")


def test_trusted_ftpfn_requires_md5_and_sha256(monkeypatch, tmp_path):
    import hashlib

    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"verified bytes")
    monkeypatch.setattr(
        ftpfn_module,
        "FTPFN_ARTIFACT_MD5",
        hashlib.md5(artifact.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(ftpfn_module, "FTPFN_ARTIFACT_SHA256", "0" * 64)
    with pytest.raises(ValueError):
        TrustedFTPFN._verify_checksum(str(artifact))

    monkeypatch.setattr(
        ftpfn_module,
        "FTPFN_ARTIFACT_SHA256",
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )
    TrustedFTPFN._verify_checksum(str(artifact))


def test_trusted_ftpfn_places_all_prediction_tensors_on_requested_device(monkeypatch):
    recorded_devices = []
    real_tensor = torch.tensor

    def recording_tensor(*args, **kwargs):
        recorded_devices.append(kwargs.get("device"))
        return real_tensor(*args, **kwargs)

    class Curve:
        def __init__(self, **kwargs):
            self.t = kwargs["t"]

    class Result:
        def pi(self, best):
            return real_tensor([0.5])

    class Predictor:
        def predict(self, context, query):
            return [Result() for _ in query]

    fake_ifbo = types.ModuleType("ifbo")
    fake_ifbo.Curve = Curve
    monkeypatch.setitem(sys.modules, "ifbo", fake_ifbo)
    monkeypatch.setattr(torch, "tensor", recording_tensor)

    model = object.__new__(TrustedFTPFN)
    model._device = torch.device("cpu")
    model._model = Predictor()
    model._torch = torch
    posterior_input = PosteriorInput(
        context_ids=np.array([1]),
        context_t=np.array([0.25]),
        context_hp=np.array([[0.5]]),
        context_y=np.array([0.5]),
        query_ids=np.array([1]),
        query_t=np.array([0.5]),
        query_hp=np.array([[0.5]]),
    )
    model.pi(posterior_input, threshold=0.6)
    assert recorded_devices
    assert all(device == torch.device("cpu") for device in recorded_devices)


def test_trusted_ftpfn_supports_query_only_context():
    model = object.__new__(TrustedFTPFN)
    posterior_input = PosteriorInput(
        context_ids=np.array([], dtype=np.int64),
        context_t=np.array([], dtype=np.float64),
        context_hp=np.empty((0, 1), dtype=np.float64),
        context_y=np.array([], dtype=np.float64),
        query_ids=np.array([0, 0], dtype=np.int64),
        query_t=np.array([0.25, 0.5], dtype=np.float64),
        query_hp=np.array([[0.1], [0.9]], dtype=np.float64),
    )
    assert model.pi(posterior_input, threshold=0.5) == pytest.approx([0.5, 0.5])
