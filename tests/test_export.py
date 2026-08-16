"""Tests for src.export — the artefacts the manuscript quotes from.

These files are the paper's evidence base, so the tests check that each
one is complete and self-describing: a reader must be able to tell, from
the file alone, which commit produced it and what the numbers mean.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.export import (
    TRACKED_PACKAGES,
    export_all,
    git_provenance,
    package_versions,
    write_ablation,
    write_baselines,
    write_feature_importance,
    write_model_comparison,
    write_results_summary,
    write_run_metadata,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def summary() -> pd.DataFrame:
    rows = []
    for model, roc, pr in [
        ("xgboost", 0.85, 0.60), ("lightgbm", 0.88, 0.66),
        ("logistic", 0.91, 0.70), ("majority", 0.50, 0.24),
        ("persistence", 0.80, 0.53), ("moisture_threshold", 0.88, 0.70),
    ]:
        for metric, value in [("roc_auc", roc), ("pr_auc", pr)]:
            rows.append({
                "model": model, "metric": metric, "mean": value,
                "std": 0.12, "ci_low": max(0.0, value - 0.15),
                "ci_high": min(1.0, value + 0.15),
                "seed_std": 0.0, "seed_ci_low": value, "seed_ci_high": value,
                "n_seeds": 10, "n_folds": 5,
                "deterministic_across_seeds": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def ablation_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"feature_set": s, "description": d, "model": "xgboost",
         "n_features": n, "n_rows": 1090,
         "roc_auc_mean": v, "roc_auc_std": 0.1,
         "roc_auc_ci_low": v - 0.1, "roc_auc_ci_high": v + 0.1}
        for s, d, n, v in [
            ("A", "moisture only", 16, 0.86), ("B", "+ weather", 32, 0.82),
            ("C", "+ calendar", 35, 0.82), ("D", "full", 40, 0.85),
            ("E", "weather only", 16, 0.54),
        ]
    ])


@pytest.fixture
def shap_manifest() -> dict:
    return {
        "model": "xgboost",
        "fold": 4,
        "n_features": 3,
        "mean_abs_shap": {
            "irrigation_event_lag1h": 1.70,
            "soil_moisture_lag1h": 1.20,
            "humidity_lag24h": 0.33,
        },
        "instances": {},
        "top_features": ["irrigation_event_lag1h"],
    }


@pytest.fixture
def blocks() -> dict:
    return {
        "irrigation": ["irrigation_event_lag1h"],
        "moisture": ["soil_moisture_lag1h"],
        "weather": ["humidity_lag24h"],
    }


@pytest.fixture
def comparison() -> dict:
    return {
        "metric": "roc_auc", "observed_diff": -0.0153, "p_value": 0.0074,
        "ci_low": -0.0348, "ci_high": -0.0032, "n_valid": 10000,
        "n_iterations": 10000, "block_size": 24, "seed": 42,
        "model_a": "lightgbm", "model_b": "logistic", "p_value_iid": 0.0020,
    }


# ── Provenance ───────────────────────────────────────────────────────


class TestProvenance:
    def test_reports_a_commit_in_this_repository(self) -> None:
        provenance = git_provenance()
        assert provenance["commit"] is not None
        assert len(provenance["commit"]) == 40

    def test_dirty_flag_is_a_bool_not_a_guess(self) -> None:
        provenance = git_provenance()
        assert isinstance(provenance["dirty"], bool)

    def test_dirty_tree_carries_an_explanation(self) -> None:
        """A result from uncommitted code must say so in the file."""
        provenance = git_provenance()
        if provenance["dirty"]:
            assert "not reproducible" in provenance["note"]

    def test_dirty_ignores_untracked_files(self) -> None:
        """Otherwise every run flags itself dirty.

        The run writes data/outputs/, so if those files are untracked
        they appear in `git status --porcelain` and the flag would be
        true on every clean checkout — describing the artefacts instead
        of the code that produced them.
        """
        import subprocess

        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        assert git_provenance()["dirty"] == bool(tracked)

    def test_untracked_count_is_reported_separately(self) -> None:
        """Hidden, not ignored: the count is still recorded."""
        provenance = git_provenance()
        assert isinstance(provenance["untracked_files"], int)
        assert provenance["untracked_files"] >= 0

    def test_tracks_every_version_affecting_numbers(self) -> None:
        versions = package_versions()
        assert "python" in versions
        for package in TRACKED_PACKAGES:
            assert package in versions

    def test_core_packages_resolve_to_versions(self) -> None:
        versions = package_versions()
        for package in ("numpy", "pandas", "sklearn", "xgboost"):
            assert versions[package] is not None


# ── Individual writers ───────────────────────────────────────────────


class TestResultsSummary:
    def test_has_the_required_columns(self, summary, tmp_path) -> None:
        path = write_results_summary(summary, tmp_path)
        frame = pd.read_csv(path)
        assert {"model", "metric", "mean", "std", "ci_low", "ci_high"} <= set(
            frame.columns
        )

    def test_preserves_the_seed_columns(self, summary, tmp_path) -> None:
        """The degenerate seed interval is kept, just not headlined."""
        frame = pd.read_csv(write_results_summary(summary, tmp_path))
        assert {"seed_std", "seed_ci_low", "seed_ci_high"} <= set(frame.columns)
        assert "deterministic_across_seeds" in frame.columns

    def test_every_row_carries_an_interval(self, summary, tmp_path) -> None:
        frame = pd.read_csv(write_results_summary(summary, tmp_path))
        assert frame["ci_low"].notna().all()
        assert frame["ci_high"].notna().all()

    def test_sorted_best_first_within_metric(self, summary, tmp_path) -> None:
        frame = pd.read_csv(write_results_summary(summary, tmp_path))
        for _, block in frame.groupby("metric"):
            assert block["mean"].is_monotonic_decreasing


class TestBaselines:
    def test_contains_only_baselines(self, summary, tmp_path) -> None:
        names = ["majority", "persistence", "moisture_threshold", "logistic"]
        frame = pd.read_csv(write_baselines(
            summary, tmp_path, baseline_names=names,
            main_model_names=["xgboost", "lightgbm"], positive_rate=0.236,
        ))
        assert set(frame["model"]) <= set(names)

    def test_records_the_no_skill_reference(self, summary, tmp_path) -> None:
        """PR-AUC must never be read against 0.5."""
        frame = pd.read_csv(write_baselines(
            summary, tmp_path,
            baseline_names=["majority", "persistence"],
            main_model_names=["xgboost"], positive_rate=0.236,
        ))
        pr = frame[frame["metric"] == "pr_auc"]
        assert np.allclose(pr["no_skill_reference"], 0.236)
        roc = frame[frame["metric"] == "roc_auc"]
        assert np.allclose(roc["no_skill_reference"], 0.5)

    def test_delta_shows_a_baseline_beating_the_main_model(
        self, summary, tmp_path,
    ) -> None:
        """The finding this column exists to surface."""
        frame = pd.read_csv(write_baselines(
            summary, tmp_path,
            baseline_names=["moisture_threshold"],
            main_model_names=["xgboost", "lightgbm"], positive_rate=0.236,
        ))
        pr = frame[frame["metric"] == "pr_auc"].iloc[0]
        assert pr["delta_vs_best_main"] > 0


class TestAblationExport:
    def test_row_per_set(self, ablation_summary, tmp_path) -> None:
        frame = pd.read_csv(write_ablation(ablation_summary, tmp_path))
        assert set(frame["feature_set"]) == {"A", "B", "C", "D", "E"}

    def test_exports_row_count_for_verification(
        self, ablation_summary, tmp_path,
    ) -> None:
        """A reader must be able to confirm the sets shared their rows."""
        frame = pd.read_csv(write_ablation(ablation_summary, tmp_path))
        assert frame["n_rows"].nunique() == 1


class TestFeatureImportance:
    def test_ranked_descending(self, shap_manifest, tmp_path) -> None:
        frame = pd.read_csv(write_feature_importance(shap_manifest, tmp_path))
        assert frame["mean_abs_shap"].is_monotonic_decreasing
        assert frame["rank"].tolist() == [1, 2, 3]

    def test_tags_each_feature_with_its_block(
        self, shap_manifest, blocks, tmp_path,
    ) -> None:
        frame = pd.read_csv(
            write_feature_importance(shap_manifest, tmp_path, blocks=blocks)
        )
        assert frame.loc[0, "block"] == "irrigation"
        assert frame["block"].notna().all()

    def test_records_which_model_and_fold(
        self, shap_manifest, tmp_path,
    ) -> None:
        frame = pd.read_csv(write_feature_importance(shap_manifest, tmp_path))
        assert (frame["model"] == "xgboost").all()
        assert (frame["fold"] == 4).all()


class TestModelComparison:
    def test_records_p_value_and_interval(self, comparison, tmp_path) -> None:
        path = write_model_comparison(comparison, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["available"] is True
        assert payload["p_value"] == pytest.approx(0.0074)
        assert payload["ci_low"] < payload["ci_high"]

    def test_interprets_an_interval_excluding_zero(
        self, comparison, tmp_path,
    ) -> None:
        payload = json.loads(
            write_model_comparison(comparison, tmp_path).read_text()
        )
        assert "excludes zero" in payload["interpretation"]

    def test_interprets_an_interval_spanning_zero(
        self, comparison, tmp_path,
    ) -> None:
        comparison = {**comparison, "ci_low": -0.02, "ci_high": 0.03}
        payload = json.loads(
            write_model_comparison(comparison, tmp_path).read_text()
        )
        assert "spans zero" in payload["interpretation"]
        assert "not statistically distinguishable" in payload["interpretation"]

    def test_explains_which_p_value_to_quote(
        self, comparison, tmp_path,
    ) -> None:
        payload = json.loads(
            write_model_comparison(comparison, tmp_path).read_text()
        )
        assert "moving-block" in payload["p_value_note"]

    def test_absent_comparison_is_recorded_not_omitted(self, tmp_path) -> None:
        payload = json.loads(
            write_model_comparison(None, tmp_path).read_text()
        )
        assert payload["available"] is False
        assert "reason" in payload


class TestRunMetadata:
    @pytest.fixture
    def metadata(self, tmp_path) -> dict:
        from src.config import build_loader_config

        path = write_run_metadata(
            tmp_path, n_rows=1313, n_features=40, positive_rate=0.2361,
            seeds=range(10), n_folds=5,
            dataset_start="2022-07-13 04:00:00",
            dataset_end="2022-09-17 03:00:00",
            loader_config=build_loader_config("configs/default.yaml"),
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_records_dataset_shape_and_balance(self, metadata) -> None:
        assert metadata["dataset"]["n_rows"] == 1313
        assert metadata["dataset"]["n_features"] == 40
        assert metadata["dataset"]["positive_rate"] == pytest.approx(0.2361)

    def test_records_seeds_and_protocol(self, metadata) -> None:
        assert metadata["protocol"]["seeds"] == list(range(10))
        assert metadata["protocol"]["n_folds"] == 5
        assert "no shuffle" in metadata["protocol"]["validation"]

    def test_records_library_versions(self, metadata) -> None:
        for package in ("numpy", "pandas", "xgboost", "shap"):
            assert package in metadata["environment"]

    def test_records_git_commit(self, metadata) -> None:
        assert metadata["git"]["commit"] is not None

    def test_records_the_site_so_geography_travels_with_results(
        self, metadata,
    ) -> None:
        """The defect that started this rewrite must be visible here."""
        site = metadata["site"]
        assert site["country"] == "Paraguay"
        assert -26.0 < site["latitude"] < -24.0
        assert site["mendeley_utc_offset_hours"] == -4.0

    def test_records_generation_timestamp(self, metadata) -> None:
        assert metadata["generated_utc"].startswith("20")


# ── Everything together ──────────────────────────────────────────────


class TestExportAll:
    def test_writes_every_artefact(
        self, summary, ablation_summary, shap_manifest, blocks, comparison,
        tmp_path,
    ) -> None:
        fold_table = pd.DataFrame({
            "fold": [1, 2], "n_train": [223, 441], "n_test": [218, 218],
            "test_positive_rate": [0.03, 0.11],
        })
        paths = export_all(
            tmp_path,
            summary=summary,
            ablation_summary=ablation_summary,
            fold_table=fold_table,
            shap_manifest=shap_manifest,
            comparison=comparison,
            blocks=blocks,
            baseline_names=["majority", "persistence", "logistic",
                            "moisture_threshold"],
            main_model_names=["xgboost", "lightgbm"],
            n_rows=1313, n_features=40, positive_rate=0.2361,
            seeds=range(10), n_folds=5,
            dataset_start="2022-07-13", dataset_end="2022-09-17",
        )
        expected = {
            "results_summary", "baselines", "ablation", "feature_importance",
            "folds", "model_comparison", "run_metadata",
        }
        assert set(paths) == expected
        for name, path in paths.items():
            assert path.exists(), name
            assert path.stat().st_size > 0, name
