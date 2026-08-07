import subprocess
import sys
import textwrap


def test_experiment_modules_import_without_bettermap():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockBettermap(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "bettermap":
                    raise ImportError("bettermap is unavailable")
                return None

        if sys.platform == "win32":
            sys.meta_path.insert(0, BlockBettermap())

        import olmo_core
        import olmo_core.data
        import scripts.train.engram_experiment.common

        assert not hasattr(sys.modules["olmo_core.data.data_loader"], "bettermap")
        if sys.platform == "win32":
            assert "bettermap" in sys.modules
            try:
                sys.modules["bettermap"].ordered_map_per_thread(())
            except RuntimeError as error:
                assert "unavailable on Windows" in str(error)
            else:
                raise AssertionError("Windows bettermap fallback unexpectedly executed")
        else:
            assert "bettermap" in sys.modules
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
