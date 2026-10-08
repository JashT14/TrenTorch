import argparse
from unittest.mock import MagicMock

from platforms.cli.core.config import CLIConfig
from platforms.cli.processes.benchmark import BenchmarkCommand


def test_benchmark_subcommands_parser_supports_baseline_and_run_alias():
    """Verify that the benchmark argument parser accepts both 'baseline' and 'run'."""
    config = CLIConfig.from_project_root()
    cmd = BenchmarkCommand(config)

    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)

    # 'baseline' subcommand
    args_baseline = parser.parse_args(["baseline"])
    assert args_baseline.benchmark_command == "baseline"

    # 'run' alias
    args_run = parser.parse_args(["run"])
    assert args_run.benchmark_command == "run"

    # 'capstone' subcommand
    args_capstone = parser.parse_args(["capstone"])
    assert args_capstone.benchmark_command == "capstone"


def test_benchmark_run_dispatches_to_run_baseline():
    """Verify that both 'baseline' and 'run' route to _run_baseline."""
    config = CLIConfig.from_project_root()
    cmd = BenchmarkCommand(config)

    cmd._run_baseline = MagicMock(return_value=0)
    cmd._run_capstone = MagicMock(return_value=0)

    # baseline
    res_baseline = cmd.run(argparse.Namespace(benchmark_command="baseline"))
    assert res_baseline == 0
    cmd._run_baseline.assert_called_once()

    cmd._run_baseline.reset_mock()

    # run alias
    res_run = cmd.run(argparse.Namespace(benchmark_command="run"))
    assert res_run == 0
    cmd._run_baseline.assert_called_once()
    cmd._run_capstone.assert_not_called()
