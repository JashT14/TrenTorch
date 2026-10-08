"""
Regression test for Issue 5: State loss and working directory drift due to
hardcoded relative Path("user_data").

Verifies that Milestone, Benchmark, and Reset operations always anchor to
config.project_root / "user_data" regardless of current working directory.
"""

import json
from argparse import Namespace
from io import StringIO
from pathlib import Path

from rich.console import Console

from platforms.cli.cli_platform.package.reset import ResetCommand
from platforms.cli.core.config import CLIConfig, get_user_data_dir
from platforms.cli.processes.benchmark import BenchmarkCommand
from platforms.cli.processes.milestone.system import (
    MilestoneSystem,
    _load_completed_module_numbers,
)


def test_get_user_data_dir_anchors_to_project_root(tmp_path):
    """Verify get_user_data_dir returns user_data under project_root."""
    user_data = get_user_data_dir(tmp_path)
    assert user_data == tmp_path / "user_data"
    assert user_data.is_dir()


def test_benchmark_baseline_anchors_to_project_root_when_cwd_is_subdirectory(tmp_path, monkeypatch):
    """Verify benchmark baseline writes to project_root/user_data, not cwd/user_data."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")

    sub_dir = project_root / "data" / "modules"
    sub_dir.mkdir(parents=True)
    monkeypatch.chdir(sub_dir)

    config = CLIConfig.from_project_root(project_root)
    cmd = BenchmarkCommand(config)
    cmd.console = Console(file=StringIO(), width=120, no_color=True)

    # Mock internal op benchmarks to return quickly
    monkeypatch.setattr(cmd, "_benchmark_tensor_ops", lambda: 0.001)
    monkeypatch.setattr(cmd, "_benchmark_matmul", lambda: 0.001)
    monkeypatch.setattr(cmd, "_benchmark_forward_pass", lambda: 0.001)

    result = cmd.run(Namespace(benchmark_command="baseline"))
    assert result == 0

    # Ensure saved in project_root/user_data/benchmarks
    root_benchmarks = project_root / "user_data" / "benchmarks"
    assert root_benchmarks.exists()
    assert len(list(root_benchmarks.glob("baseline_*.json"))) == 1

    # Ensure no orphan user_data in cwd (sub_dir)
    assert not (sub_dir / "user_data").exists()


def test_milestone_system_reads_progress_from_project_root_from_any_cwd(tmp_path, monkeypatch):
    """Verify milestone progress checks find project_root/user_data when cwd is elsewhere."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    user_data = project_root / "user_data"
    user_data.mkdir()
    (user_data / "progress.json").write_text(
        json.dumps({"completed_modules": ["01", "02"]}), encoding="utf-8"
    )

    other_dir = tmp_path / "somewhere_else"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    config = CLIConfig.from_project_root(project_root)
    completed_nums = _load_completed_module_numbers(project_root)
    assert 1 in completed_nums
    assert 2 in completed_nums

    system = MilestoneSystem(config)
    assert system._is_module_completed("01") is True
    assert system._is_module_completed("02") is True
    assert system._is_module_completed("03") is False

    # Ensure no orphan user_data created in other_dir
    assert not (other_dir / "user_data").exists()


def test_package_reset_modifies_project_root_user_data_when_cwd_is_nested(tmp_path, monkeypatch):
    """Verify reset commands target project_root/user_data, not cwd/user_data."""
    project_root = tmp_path / "repo"
    project_root.mkdir()
    user_data = project_root / "user_data"
    user_data.mkdir()
    progress_file = user_data / "progress.json"
    progress_file.write_text(
        json.dumps({"completed_modules": ["01"]}), encoding="utf-8"
    )

    nested_dir = project_root / "nested" / "dir"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    config = CLIConfig.from_project_root(project_root)
    cmd = ResetCommand(config)
    cmd.console = Console(file=StringIO(), width=120, no_color=True)

    result = cmd._reset_progress(Namespace(force=True, backup=False))
    assert result == 0

    # Ensure progress in project_root was reset
    progress_after = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress_after.get("completed_modules") == []

    # Ensure no orphan user_data in nested_dir
    assert not (nested_dir / "user_data").exists()
