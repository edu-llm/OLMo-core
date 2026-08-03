"""
Tests for the entry points under ``src/scripts``.

Every other directory in ``src/test`` is a package; this one was the gap, which left its test
modules resolving as top-level names and put their own directory on ``sys.path``. That is fine
while the directory holds only loose ``*_test.py`` files, and stops being fine as soon as a
subdirectory shares a name with the code it tests -- ``factcrowd`` shadowing
``src/scripts/train/factcrowd`` being the case that surfaced it.
"""
