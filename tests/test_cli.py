import os
import re
import subprocess
import sysconfig
from importlib.metadata import version as package_version
from pathlib import Path

from typer.testing import CliRunner

import anila
from anila import (
    evaluate_benchmark_suite,
    export_safetensors_checkpoint,
    generate_text,
    sample_text,
    stream_text,
    train,
    train_byte_bpe,
)
from anila.benchmark import evaluate_benchmark_suite as module_evaluate_benchmark_suite
from anila.checkpoint import export_safetensors_checkpoint as module_export_safetensors_checkpoint
from anila.cli import app
from anila.sampling import generate_text as module_generate_text
from anila.sampling import sample_text as module_sample_text
from anila.sampling import stream_text as module_stream_text
from anila.tokenization import train_byte_bpe as module_train_byte_bpe
from anila.training import train as module_train

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_cli_output(output: str) -> str:
    return ANSI_ESCAPE_RE.sub("", output)


def test_cli_version_reports_installed_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"anila {package_version('anila')}"


def test_installed_command_version_uses_stdout_without_stderr() -> None:
    # Installed command smoke: exercise the console script, not only Typer's in-process runner.
    script_name = "anila.exe" if os.name == "nt" else "anila"
    command = Path(sysconfig.get_path("scripts")) / script_name
    assert command.exists()

    result = subprocess.run([str(command), "--version"], text=True, capture_output=True, check=False)

    assert result.returncode == 0
    assert result.stdout.strip() == f"anila {package_version('anila')}"
    assert result.stderr == ""


def test_cli_help_lists_resource_groups_and_version() -> None:
    result = CliRunner().invoke(app, ["--help"])
    output = _plain_cli_output(result.output)

    assert result.exit_code == 0
    assert "--version" in output
    assert "tokenizer" in output
    assert "model" in output
    assert "checkpoint" in output


def test_cli_generate_logprobs_requires_json(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "model",
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--prompt",
            "Anila is",
            "--logprobs",
        ],
    )
    output = _plain_cli_output(result.output)

    assert result.exit_code != 0
    assert "--logprobs requires --json" in output


def test_top_level_api_exports_common_entrypoints() -> None:
    assert anila.__version__ == package_version("anila")
    assert evaluate_benchmark_suite is module_evaluate_benchmark_suite
    assert export_safetensors_checkpoint is module_export_safetensors_checkpoint
    assert generate_text is module_generate_text
    assert sample_text is module_sample_text
    assert stream_text is module_stream_text
    assert train is module_train
    assert train_byte_bpe is module_train_byte_bpe
