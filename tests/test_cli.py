from importlib.metadata import version as package_version

from typer.testing import CliRunner

import anila
from anila import generate_text, sample_text, stream_text, train, train_byte_bpe
from anila.cli import app
from anila.sampling import generate_text as module_generate_text
from anila.sampling import sample_text as module_sample_text
from anila.sampling import stream_text as module_stream_text
from anila.tokenization import train_byte_bpe as module_train_byte_bpe
from anila.training import train as module_train


def test_cli_version_reports_installed_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"anila {package_version('anila')}"


def test_cli_help_lists_resource_groups_and_version() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.output
    assert "tokenizer" in result.output
    assert "model" in result.output
    assert "checkpoint" in result.output


def test_top_level_api_exports_common_entrypoints() -> None:
    assert anila.__version__ == package_version("anila")
    assert generate_text is module_generate_text
    assert sample_text is module_sample_text
    assert stream_text is module_stream_text
    assert train is module_train
    assert train_byte_bpe is module_train_byte_bpe
