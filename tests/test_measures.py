"""Unit tests for all 5 measures."""


class TestPropensityAUC:
    def test_returns_valid_range(self, small_train_test, small_synthetic):
        from tsd.measures.propensity_auc import propensity_auc

        df_train, _ = small_train_test
        result = propensity_auc(df_train, small_synthetic)
        assert "auc" in result
        assert 0.0 <= result["auc"] <= 1.0

    def test_identical_data_returns_auc(self, small_train_test):
        from tsd.measures.propensity_auc import propensity_auc

        df_train, _ = small_train_test
        result = propensity_auc(df_train, df_train.copy())
        # With identical data, AUC should still be a valid number in [0, 1]
        assert 0.0 <= result["auc"] <= 1.0


class TestDCRPrivacy:
    def test_returns_valid_range(self, small_train_test, small_synthetic):
        from tsd.measures.dcr_privacy import dcr_privacy

        df_train, _ = small_train_test
        result = dcr_privacy(df_train, small_synthetic)
        assert "dcr_percentile" in result
        assert result["dcr_percentile"] >= 0

    def test_identical_data_low_dcr(self, small_train_test):
        from tsd.measures.dcr_privacy import dcr_privacy

        df_train, _ = small_train_test
        result = dcr_privacy(df_train, df_train.copy())
        # Identical → distances are 0 → very low DCR
        assert result["dcr_percentile"] < 0.01


class TestMembershipInference:
    def test_returns_valid_keys(self, small_train_test, small_synthetic):
        from tsd.measures.membership_inference import membership_inference_privacy

        df_train, df_test = small_train_test
        result = membership_inference_privacy(df_train, df_test, small_synthetic)
        assert "attack_auc" in result
        assert "attack_valid" in result
        assert 0.0 <= result["attack_auc"] <= 1.0


class TestTSTRUtility:
    def test_returns_f1_ratio(self, small_train_test, small_synthetic):
        from tsd.measures.tstr_utility import tstr_utility

        df_train, df_test = small_train_test
        result = tstr_utility(df_train, small_synthetic, df_test, target_col="target")
        assert "f1_ratio" in result
        assert result["f1_ratio"] >= 0

    def test_perfect_copy_high_ratio(self, small_train_test):
        from tsd.measures.tstr_utility import tstr_utility

        df_train, df_test = small_train_test
        # Synthetic = copy of real → should get high utility ratio
        result = tstr_utility(df_train, df_train.copy(), df_test, target_col="target")
        assert result["f1_ratio"] >= 0.8


class TestTSTRUtilityMLP:
    def test_returns_f1_ratio(self, small_train_test, small_synthetic):
        from tsd.measures.tstr_utility import tstr_utility_mlp

        df_train, df_test = small_train_test
        result = tstr_utility_mlp(df_train, small_synthetic, df_test, target_col="target")
        assert "f1_ratio" in result
        assert result["f1_ratio"] >= 0
        assert result["evaluator"] == "mlp"

    def test_perfect_copy_high_ratio(self, small_train_test):
        from tsd.measures.tstr_utility import tstr_utility_mlp

        df_train, df_test = small_train_test
        result = tstr_utility_mlp(df_train, df_train.copy(), df_test, target_col="target")
        assert result["f1_ratio"] >= 0.5  # MLP may differ slightly from GBT

    def test_returns_all_expected_keys(self, small_train_test, small_synthetic):
        from tsd.measures.tstr_utility import tstr_utility_mlp

        df_train, df_test = small_train_test
        result = tstr_utility_mlp(df_train, small_synthetic, df_test, target_col="target")
        expected_keys = {
            "f1_real",
            "f1_synthetic",
            "f1_ratio",
            "auc_real",
            "auc_synthetic",
            "auc_ratio",
            "utility_score",
            "evaluator",
        }
        assert set(result.keys()) == expected_keys


class TestFairnessGap:
    def test_returns_max_gap(self, small_train_test, small_synthetic):
        from tsd.measures.fairness_gap import fairness_gap

        df_train, df_test = small_train_test
        result = fairness_gap(
            df_train,
            small_synthetic,
            df_test,
            target_col="target",
            subgroup_col="race",
            min_subgroup_size=5,
        )
        assert "max_gap" in result
        assert result["max_gap"] >= 0
