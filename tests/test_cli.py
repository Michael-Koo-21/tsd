"""Integration tests for CLI commands."""

import subprocess
import sys


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
