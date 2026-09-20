"""Regression test for loading many shards under a low file-descriptor limit."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    import resource
except ImportError:
    resource = None


class ShardLoadingTests(unittest.TestCase):
    @unittest.skipIf(resource is None, "Requires Unix file-descriptor limits")
    def test_many_shards_under_low_file_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            shards = run / "collection" / "shards"
            shards.mkdir(parents=True)
            rows = [{"id": str(i)} for i in range(80)]
            (run / "effective-plan.json").write_text(json.dumps({"configuration": {}}))
            (run / "stimuli.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            for i, row in enumerate(rows):
                np.savez(shards / f"{i:07d}.npz", ids=np.array([row["id"]]),
                         probabilities=np.ones((1, 9))/9, layer_27=np.full((1, 4), i))
            code = f"""
import sys, resource
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from _common import load_run, activations
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(32, soft), hard))
cfg, rows, probabilities, pieces = load_run({str(run)!r})
assert probabilities.shape == (80, 9)
assert len(rows) == len(pieces) == 80
assert all(isinstance(p, dict) for p in pieces)
x = activations(pieces, 27)
assert x.shape == (80, 4) and x[-1, 0] == 79
with open({str(run / 'after-load.txt')!r}, 'w') as f:
    f.write('File descriptors remain available')
"""
            completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
