import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class PortabilityTests(unittest.TestCase):
    def test_legacy_python_shebangs_are_generic(self):
        legacy_files = list((ROOT / "scripts" / "legacy").rglob("*.py"))
        self.assertTrue(legacy_files)
        for path in legacy_files:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("/opt/anaconda3/envs/medmatch/bin/python3", first_line, path.as_posix())


if __name__ == "__main__":
    unittest.main()
