"""Unit tests for SHAP and gain overfit filters."""

import pandas as pd

from fs_xgb.fs_logic.fs_pipeline import (
    _apply_selection_rules,
    _flag_gain_overfit_features,
    _flag_overfit_features,
)


def _make_fi_table():
    return pd.DataFrame(
        {
            "feature": ["a", "b"],
            "delta_mean": [0.01, 0.0001],
            "delta_std": [0.001, 0.0002],
        }
    )


def _make_shap_series():
    return pd.Series({"a": 0.5, "b": 0.4, "c": 0.3})


def _make_gain_series():
    return pd.Series({"a": 0.2, "b": 0.1, "c": 0.05})


def test_flag_overfit_features_respects_thresholds():
    fi_table = _make_fi_table()
    shap_importance = _make_shap_series()
    config = {
        "enabled": True,
        "max_shap_rank": 2,
        "delta_max": 0.0005,
        "delta_std_max": 0.0005,
    }
    flagged = _flag_overfit_features(fi_table, shap_importance, config)
    assert flagged == ["b"]


def test_apply_selection_rules_drops_overfit_features():
    fi_table = _make_fi_table()
    shap_importance = _make_shap_series()
    gain_importance = _make_gain_series()
    config = {
        "thresholds": {"delta_abs_min": 0.0, "k_noise_std": 0.0},
        "rest_policy": "drop_all",
        "drop_negative_features": False,
        "overfit_filter": {
            "enabled": True,
            "max_shap_rank": 3,
            "delta_max": 0.0005,
            "delta_std_max": 0.0005,
            "action": "drop",
        },
    }
    keep, drop, shap_flags, gain_flags, threshold = _apply_selection_rules(
        fi_table, shap_importance, gain_importance, config
    )
    assert keep == ["a"]
    assert "b" in drop
    assert shap_flags == ["b", "c"]
    assert gain_flags == []


def test_apply_selection_rules_demote_pushes_feature_into_rest_pool():
    fi_table = _make_fi_table()
    shap_importance = _make_shap_series()
    gain_importance = _make_gain_series()
    config = {
        "thresholds": {"delta_abs_min": 0.0, "k_noise_std": 0.0},
        "rest_policy": "keep_above_min_shap",
        "rest_min_shap_rank": 2,
        "drop_negative_features": False,
        "overfit_filter": {
            "enabled": True,
            "max_shap_rank": 3,
            "delta_max": 0.0005,
            "delta_std_max": 0.0005,
            "action": "demote",
        },
    }
    keep, drop, shap_flags, gain_flags, threshold = _apply_selection_rules(
        fi_table, shap_importance, gain_importance, config
    )
    # Feature "b" initially qualifies via permutation but is demoted. Since the
    # rest policy keeps SHAP ranks <=2, it is re-added via the rest bucket.
    assert set(keep) == {"a", "b"}
    assert shap_flags == ["b", "c"]
    assert gain_flags == []


def test_flag_gain_overfit_features_respects_thresholds():
    fi_table = _make_fi_table()
    gain_importance = _make_gain_series()
    config = {
        "enabled": True,
        "max_gain_rank": 2,
        "delta_max": 0.0005,
        "delta_std_max": 0.0005,
    }
    flagged = _flag_gain_overfit_features(fi_table, gain_importance, config)
    assert flagged == ["b"]
