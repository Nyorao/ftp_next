"""Lets a bare `pytest` (not only `python -m pytest`) import repository packages.

pytest inserts the directory of a root-level conftest.py into sys.path, so
`freqssl`, `model`, `config` and the other top-level modules resolve the same
way they do under `python -m pytest`.
"""
