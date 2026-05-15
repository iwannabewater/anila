from pathlib import Path

import pytest

from anila.training import CheckpointManager


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
