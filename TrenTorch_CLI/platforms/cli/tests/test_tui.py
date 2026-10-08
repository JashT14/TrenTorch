import asyncio

import pytest

from platforms.cli.core.config import CLIConfig
from platforms.cli.tui.command import TUICommand

# `textual` is an optional extra (`trentorch[tui]`); skip the app tests when
# it is not installed rather than erroring at import time.
pytest.importorskip("textual")

from platforms.cli.tui.app import TrenTorchApp  # noqa: E402  (after importorskip)


def test_tui_app_initialization():
    """Verify that the TUI app initializes, populates modules, and sets up tabs."""

    async def _runner():
        config = CLIConfig.from_project_root()
        app = TrenTorchApp(config=config, initial_module="01")

        async with app.run_test() as pilot:
            # Check title
            assert "Tren⚡️Torch" in app.TITLE

            # Check tabs exist
            tabs = app.query_one("#main-tabs")
            assert tabs is not None

            # Verify default active tab is modules
            assert tabs.active == "modules-tab"

            # Verify switching tabs via bindings
            await pilot.press("2")
            assert tabs.active == "milestones-tab"

            await pilot.press("3")
            assert tabs.active == "benchmarks-tab"

            await pilot.press("4")
            assert tabs.active == "health-tab"

            await pilot.press("1")
            assert tabs.active == "modules-tab"

    asyncio.run(_runner())


def test_tui_command_registration():
    """Verify TUICommand exposes the expected metadata."""
    config = CLIConfig.from_project_root()
    cmd = TUICommand(config)
    assert cmd.name == "tui"
    assert "interactive" in cmd.description.lower()


def test_tui_run_benchmarks_button_triggers_baseline(monkeypatch):
    """Verify that clicking the benchmark action button invokes 'benchmark baseline'."""

    async def _runner():
        config = CLIConfig.from_project_root()
        app = TrenTorchApp(config=config, initial_module="01")
        spawned = []

        def fake_subprocess_worker(cmd, log_widget_id="execution-log", title="Task"):
            spawned.append((cmd, log_widget_id, title))

        monkeypatch.setattr(app, "_run_subprocess_worker", fake_subprocess_worker)

        async with app.run_test() as pilot:
            btn = app.query_one("#btn-run-benchmarks")
            btn.press()
            await pilot.pause()
            assert len(spawned) == 1
            cmd, log_id, title = spawned[0]
            assert cmd[-2:] == ["benchmark", "baseline"]
            assert log_id == "benchmark-log"
            assert title == "Op Benchmarking"

    asyncio.run(_runner())

