"""Integration tests for CLI commands."""

import subprocess
import sys
from argparse import Namespace

import pytest


class TestCLI:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "tsd.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "recommend" in result.stdout

    def test_recommend_produces_output(self, tmp_path, sample_results):
        # Write sample results to a temp file
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tsd.cli",
                "recommend",
                "--results",
                str(results_file),
                "--profile",
                "balanced",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "RECOMMENDATION" in result.stdout

    def test_verify_produces_output(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tsd.cli",
                "verify",
                "--results",
                str(results_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "VERIFICATION" in result.stdout


class TestCmdRecommendDirect:
    """Direct Python calls to cmd_recommend for coverage."""

    def test_cmd_recommend_balanced(self, tmp_path, sample_results, capsys):
        from tsd.cli import cmd_recommend

        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        args = Namespace(
            results=str(results_file),
            profile="balanced",
            weights=None,
        )
        cmd_recommend(args)
        captured = capsys.readouterr()
        assert "RECOMMENDATION" in captured.out

    def test_cmd_recommend_custom_weights(self, tmp_path, sample_results, capsys):
        from tsd.cli import cmd_recommend

        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        args = Namespace(
            results=str(results_file),
            profile="balanced",
            weights='{"fidelity":0.2,"privacy":0.2,"utility":0.2,"fairness":0.2,"efficiency":0.2}',
        )
        cmd_recommend(args)
        captured = capsys.readouterr()
        assert "Custom" in captured.out

    def test_cmd_recommend_missing_file(self, tmp_path):
        from tsd.cli import cmd_recommend

        args = Namespace(
            results=str(tmp_path / "nonexistent.csv"),
            profile="balanced",
            weights=None,
        )
        with pytest.raises(SystemExit):
            cmd_recommend(args)

    def test_cmd_recommend_bad_profile(self, tmp_path, sample_results):
        from tsd.cli import cmd_recommend

        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        args = Namespace(
            results=str(results_file),
            profile="nonexistent_profile",
            weights=None,
        )
        with pytest.raises(SystemExit):
            cmd_recommend(args)


class TestCmdVerifyDirect:
    """Direct Python calls to cmd_verify for coverage."""

    def test_cmd_verify(self, tmp_path, sample_results, capsys):
        from tsd.cli import cmd_verify

        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)

        args = Namespace(results=str(results_file))
        cmd_verify(args)
        captured = capsys.readouterr()
        assert "VERIFICATION" in captured.out

    def test_cmd_verify_missing_file(self, tmp_path):
        from tsd.cli import cmd_verify

        args = Namespace(results=str(tmp_path / "nonexistent.csv"))
        with pytest.raises(SystemExit):
            cmd_verify(args)
