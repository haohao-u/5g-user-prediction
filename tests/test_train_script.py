import subprocess
import sys
from pathlib import Path


def test_train_script_can_be_invoked_directly():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "train_models.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "Train 5G user prediction models" in result.stdout
