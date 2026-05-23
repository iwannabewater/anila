import random
from pathlib import Path

import numpy as np
import pytest
import torch

from anila.training import CheckpointManager, capture_rng_state, restore_rng_state


def _step_files(root: Path) -> list[str]:
    return sorted(path.name for path in root.glob("step_*.pt"))


def test_checkpoint_manager_prunes_old_step_checkpoints(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=2)

    for step in range(1, 5):
        manager.save({"step": step}, step=step)

    root = tmp_path / "checkpoints"
    assert _step_files(root) == ["step_00000003.pt", "step_00000004.pt"]
    assert (root / "latest.pt").exists()


def test_checkpoint_manager_prunes_old_adapter_checkpoints(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, keep_last=2)

    for step in range(1, 5):
        manager.save_adapter({"step": step}, step=step)

    root = tmp_path / "checkpoints" / "adapters"
    assert _step_files(root) == ["step_00000003.pt", "step_00000004.pt"]
    assert (root / "latest.pt").exists()


def test_checkpoint_manager_validates_retention_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        CheckpointManager(tmp_path, keep_last=0)
    with pytest.raises(ValueError, match="keep_last"):
        CheckpointManager(tmp_path, keep_last=True)


def test_rng_state_round_trip_restores_all_cpu_streams() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), torch.rand(1))

    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(1))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_restore_rng_state_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError, match="rng_state"):
        restore_rng_state({"python": random.getstate(), "numpy": {}, "torch": torch.get_rng_state()})
