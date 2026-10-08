"""
Unit tests for TrenTorchStatusAnalyzer modules_path resolution and discovery.

Verifies the fix for Issue 6: TrenTorchStatusAnalyzer previously hardcoded
the non-existent data/modules/source directory, causing check_all_modules()
to silently return {} unless modules_path was manually overridden.
"""

import subprocess
from pathlib import Path

from platforms.cli.core.status_analyzer import TrenTorchStatusAnalyzer


def test_status_analyzer_points_to_data_src_when_present(tmp_path):
    """Verify modules_path resolves to data/src when present in repo_path."""
    src_dir = tmp_path / "data" / "src"
    src_dir.mkdir(parents=True)

    analyzer = TrenTorchStatusAnalyzer(repo_path=tmp_path)
    assert analyzer.modules_path == src_dir
    assert analyzer.modules_path.exists()


def test_status_analyzer_falls_back_to_data_modules_when_src_missing(tmp_path):
    """Verify modules_path falls back to data/modules when data/src is absent."""
    modules_dir = tmp_path / "data" / "modules"
    modules_dir.mkdir(parents=True)

    analyzer = TrenTorchStatusAnalyzer(repo_path=tmp_path)
    assert analyzer.modules_path == modules_dir
    assert analyzer.modules_path.exists()


def test_status_analyzer_discovers_modules_from_data_src_automatically(tmp_path, monkeypatch):
    """Verify check_all_modules finds module directories under data/src automatically."""
    src_dir = tmp_path / "data" / "src"
    (src_dir / "01_tensor").mkdir(parents=True)
    (src_dir / "01_tensor" / "01_tensor.py").write_text("x = 1\n", encoding="utf-8")
    (src_dir / "02_activations").mkdir(parents=True)
    (src_dir / "02_activations" / "02_activations.py").write_text("y = 2\n", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "SUCCESS", ""))

    analyzer = TrenTorchStatusAnalyzer(repo_path=tmp_path)
    modules = analyzer.check_all_modules()

    assert "01_tensor" in modules
    assert "02_activations" in modules
    assert len(modules) == 2


def test_status_analyzer_default_init_discovers_repo_curriculum(monkeypatch):
    """Verify initializing with no args discovers curriculum modules from active workspace."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "SUCCESS", ""))

    analyzer = TrenTorchStatusAnalyzer()
    assert analyzer.modules_path.exists()
    assert analyzer.modules_path.name == "src"
    modules = analyzer.check_all_modules()
    assert len(modules) >= 20
    assert "01_tensor" in modules
    assert "20_capstone" in modules
