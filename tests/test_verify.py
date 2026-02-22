"""Tests for tsd.analysis.verify."""

from tsd.analysis.verify import verify_claims


class TestVerifyClaims:
    def test_returns_string(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert isinstance(report, str)

    def test_contains_verification_header(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert "VERIFICATION" in report

    def test_contains_raw_means(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert "RAW MEANS" in report

    def test_contains_normalization_check(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert "NORMALIZATION CHECK" in report

    def test_contains_weighted_scores(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert "WEIGHTED SCORES" in report

    def test_contains_suspicious_results(self, tmp_path, sample_results):
        results_file = tmp_path / "results.csv"
        sample_results.to_csv(results_file, index=False)
        report = verify_claims(results_file)
        assert "SUSPICIOUS RESULTS" in report
