import pytest

from olmo_core.hpo.worker import BatchSizeMismatch, assert_resume_batch_size


def test_resume_rejects_global_batch_size_mismatch():
    # Same global batch size across a lineage is fine.
    assert_resume_batch_size(lineage_global_batch_size=1024, resumed_global_batch_size=1024)
    # A changed global batch size invalidates optimizer/data-loader state; fail closed.
    with pytest.raises(BatchSizeMismatch):
        assert_resume_batch_size(lineage_global_batch_size=1024, resumed_global_batch_size=2048)
