"""The README's headline numbers must match ``data/outputs/``.

Why this file exists
--------------------
The README once quoted nested-CV deltas of −0.001 and −0.033 while the
artefacts said +0.005 and −0.027. Nothing was wrong with the pipeline:
the README had been written against an earlier run and never revisited
after the episode-dominance rework changed those figures. A reviewer
following the data-availability link would have found three numbers that
disagreed with the manuscript.

For a study whose central claim is reproducibility, that discrepancy
reads worse than the arithmetic error it actually was — so the headline
figures are now pinned to the files that produce them. Any future run
that shifts a number fails here until the README is updated with it.

The tests parse the README rather than templating it, so the prose stays
readable and reviewable in a diff. Each failure names both the claimed
value and the artefact that contradicts it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

README = Path("README.md")
OUTPUTS = Path("data/outputs")

#: README rounds to three decimals, so allow half a unit in the last
#: place plus a margin for rounding direction.
TOL = 1e-3


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "results_summary.csv")


@pytest.fixture(scope="module")
def sensitivity() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "sensitivity_summary.csv")


def _claimed(readme: str, pattern: str) -> float:
    """Extract one number the README asserts, or fail describing the miss."""
    match = re.search(pattern, readme)
    if match is None:
        pytest.fail(
            f"README no longer contains the sentence this test pins.\n"
            f"Pattern: {pattern}\n"
            f"Either restore the claim or update the test to the new wording."
        )
    return float(match.group(1).replace("−", "-").replace("–", "-"))


def _actual_mean(results: pd.DataFrame, model: str, metric: str) -> float:
    return float(
        results[(results.model == model) & (results.metric == metric)]
        ["mean"].iloc[0]
    )


# ── Headline findings ────────────────────────────────────────────────


class TestNestedCVDeltas:
    """The claim that tuning does not rescue the tree models."""

    @staticmethod
    def _nested_delta(model: str, results: pd.DataFrame) -> float:
        nested = pd.read_csv(OUTPUTS / "nested_cv.csv")
        tuned = float(nested[nested.model == model]["pr_auc"].mean())
        return tuned - _actual_mean(results, model, "pr_auc")

    def test_xgboost_delta(self, readme, results) -> None:
        claimed = _claimed(readme, r"by ([+−-]?[0-9.]+)\s*\n?\s*\(XGBoost")
        actual = self._nested_delta("xgboost", results)
        assert claimed == pytest.approx(actual, abs=TOL), (
            f"README claims nested-CV moves XGBoost PR-AUC by {claimed:+.4f}; "
            f"nested_cv.csv and results_summary.csv give {actual:+.4f}."
        )

    def test_lightgbm_delta(self, readme, results) -> None:
        claimed = _claimed(readme, r"and ([+−-]?[0-9.]+) \(LightGBM")
        actual = self._nested_delta("lightgbm", results)
        assert claimed == pytest.approx(actual, abs=TOL), (
            f"README claims nested-CV moves LightGBM PR-AUC by {claimed:+.4f}; "
            f"the artefacts give {actual:+.4f}."
        )

    def test_sign_of_the_xgboost_delta_is_stated_correctly(
        self, readme, results,
    ) -> None:
        """A sign error here would invert the paper's argument."""
        claimed = _claimed(readme, r"by ([+−-]?[0-9.]+)\s*\n?\s*\(XGBoost")
        actual = self._nested_delta("xgboost", results)
        assert (claimed >= 0) == (actual >= 0)


class TestOnsetNoSkill:
    """The reference a reader compares the onset PR-AUC against."""

    def test_no_skill_matches_the_majority_baseline(self, readme) -> None:
        claimed = _claimed(readme, r"against a no-skill ([0-9.]+)\. The threshold")
        onset = pd.read_csv(OUTPUTS / "onset_results.csv")
        actual = float(
            onset[onset.model == "majority"]["pr_auc_mean"].iloc[0]
        )
        assert claimed == pytest.approx(actual, abs=TOL), (
            f"README quotes an onset no-skill of {claimed}; the majority "
            f"baseline in onset_results.csv scores {actual:.4f}. The "
            f"no-skill reference must be the prevalence of the rows the "
            f"metric was computed on."
        )

    def test_best_onset_score_matches(self, readme) -> None:
        claimed = _claimed(readme, r"chance: PR-AUC ([0-9.]+)")
        onset = pd.read_csv(OUTPUTS / "onset_results.csv")
        actual = float(onset["pr_auc_mean"].max())
        assert claimed == pytest.approx(actual, abs=TOL)

    def test_both_onset_rates_are_distinguished(self, readme) -> None:
        """4.5 % and 5.3 % are both in the data and mean different things.

        Quoting one without the other invites a reader who recomputes to
        conclude the README is wrong.
        """
        assert "4.5 %" in readme
        assert "5.3 %" in readme


class TestDominanceDropRange:
    """The inflation attributable to the episode-dominated fold."""

    @staticmethod
    def _drops(sensitivity: pd.DataFrame, protocol: str) -> pd.Series:
        block = sensitivity[
            (sensitivity.protocol == protocol)
            & (sensitivity.metric == "pr_auc")
            & (sensitivity.subset == "excluding_episode_dominated")
        ]
        return block["delta_vs_all_folds"].abs()

    def test_range_spans_every_model(self, readme, sensitivity) -> None:
        low = _claimed(readme, r"lowers PR-AUC by\s*\n?([0-9.]+)–[0-9.]+")
        high = _claimed(readme, r"lowers PR-AUC by\s*\n?[0-9.]+–([0-9.]+)")
        drops = self._drops(sensitivity, "main")

        assert low == pytest.approx(drops.min(), abs=TOL), (
            f"README's lower bound {low} does not match the smallest drop "
            f"{drops.min():.4f} in sensitivity_summary.csv."
        )
        assert high == pytest.approx(drops.max(), abs=TOL), (
            f"README's upper bound {high} does not match the largest drop "
            f"{drops.max():.4f}. The range must cover all six models, not "
            f"only those that happen to be discussed."
        )

    def test_range_covers_all_six_models(self, sensitivity) -> None:
        assert len(self._drops(sensitivity, "main")) == 6


class TestStableClaims:
    """Figures that were already right, pinned so they stay right."""

    def test_threshold_baseline_pr_auc(self, readme, results) -> None:
        claimed = _claimed(readme, r"threshold\s*\n?\s*baseline's ([0-9]+\.[0-9]+)")
        actual = _actual_mean(results, "moisture_threshold", "pr_auc")
        assert claimed == pytest.approx(actual, abs=TOL)

    def test_positive_rate(self, readme) -> None:
        claimed = _claimed(readme, r"positive rate \(([0-9.]+)\)")
        metadata = json.loads(
            (OUTPUTS / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert claimed == pytest.approx(
            metadata["dataset"]["positive_rate"], abs=TOL
        )

    def test_design_matrix_size(self, readme) -> None:
        metadata = json.loads(
            (OUTPUTS / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert f"{metadata['dataset']['n_rows']:,}".replace(",", " ") in readme

    def test_largest_episode_share(self, readme) -> None:
        claimed = _claimed(readme, r"supplies ([0-9.]+) %\s*\n?of all positive")
        metadata = json.loads(
            (OUTPUTS / "run_metadata.json").read_text(encoding="utf-8")
        )
        share = metadata["episode_dominance"]["largest_episode"][
            "share_of_design_matrix_positives"
        ]
        assert claimed / 100 == pytest.approx(share, abs=TOL)

    def test_dominance_table_matches_folds_csv(self, readme) -> None:
        """Every dominance value printed in §9 must come from folds.csv."""
        folds = pd.read_csv(OUTPUTS / "folds.csv")
        for _, row in folds.iterrows():
            value = row["episode_dominance"]
            if pd.isna(value):
                continue
            assert f"{value:.3f}" in readme, (
                f"fold {int(row['fold'])} has dominance {value:.3f} in "
                f"folds.csv but that value does not appear in the README "
                f"table."
            )

    def test_test_count(self, readme) -> None:
        """The advertised suite size must match what pytest collects."""
        import subprocess

        claimed = int(
            re.search(r"pytest -q\s+# ([0-9]+) tests", readme).group(1)
        )
        import sys

        collected = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--collect-only"],
            capture_output=True, text=True, check=False,
        ).stdout
        match = re.search(r"([0-9]+) tests collected", collected)
        if match is None:
            pytest.skip("could not determine the collected test count")
        assert claimed == int(match.group(1))
