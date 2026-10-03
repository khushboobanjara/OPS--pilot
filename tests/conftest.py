"""Runs before any test module: point the app at a throwaway database so tests never
touch (or get polluted by) your real opspilot.db."""
import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="opspilot_test_"))
os.environ["OPSPILOT_DATABASE_URL"] = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
