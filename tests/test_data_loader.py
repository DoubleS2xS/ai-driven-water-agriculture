"""Tests for src.data_loader — schema validation, column discovery,
merging, provenance generation, and offline mode.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DataLoaderConfig, NASAPowerConfig
from src.data_loader import (
    EXPECTED_COLUMNS,
    _identify_columns,
    load_mendeley_csv,
    merge_and_resample,
    validate_schema,
    generate_provenance,
    fetch_nasa_power,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_valid_df(n: int = 48) -> pd.DataFrame:
    """Build a minimal valid merged DataFrame for testing."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2022-07-12", periods=n, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "soil_moisture": rng.uniform(20, 80, n),
        "air_temp": rng.uniform(10, 40, n),
        "humidity": rng.uniform(20, 90, n),
        "wind_speed": rng.uniform(0, 10, n),
        "solar_radiation": rng.uniform(0, 800, n),
        "irrigation_event": rng.choice([0.0, 1.0], n),
        "flow_l": rng.uniform(0, 50, n),
        "flow_l_cumulative": np.cumsum(rng.uniform(0, 50, n)),
    })


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return _make_valid_df()


@pytest.fixture
def mendeley_moisture_file(tmp_path: Path) -> Path:
    """Create a small D_moisture.txt-like file."""
    content = textwrap.dedent("""\
        moisture,time_stamp
        74.29,12-Jul-2022 00:03:51
        73.47,12-Jul-2022 00:23:46
        72.65,12-Jul-2022 00:43:41
        71.50,12-Jul-2022 01:03:35
        70.10,12-Jul-2022 01:23:30
    """)
    f = tmp_path / "D_moisture.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def mendeley_valve_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        relay,time_stamp
        0,12-Jul-2022 00:00:00
        1,12-Jul-2022 00:05:00
        0,12-Jul-2022 00:10:00
        1,12-Jul-2022 00:15:00
        0,12-Jul-2022 00:20:00
    """)
    f = tmp_path / "D_valve.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def mendeley_flow_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        litres,time_stamp
        0,12-Jul-2022 00:00:15
        10.5,12-Jul-2022 00:05:15
        21.0,12-Jul-2022 00:10:15
        31.5,12-Jul-2022 00:15:15
        42.0,12-Jul-2022 00:20:15
    """)
    f = tmp_path / "D_flowmeter.txt"
    f.write_text(content, encoding="utf-8")
    return f


# ── Column discovery ──────────────────────────────────────────────────


class TestIdentifyColumns:
    def test_two_columns(self) -> None:
        df = pd.DataFrame({
            "moisture": [74.29, 73.47],
            "time_stamp": ["12-Jul-2022 00:03:51", "12-Jul-2022 00:23:46"],
        })
        ts_col, val_col = _identify_columns(df)
        assert ts_col == "time_stamp"
        assert val_col == "moisture"

    def test_wrong_column_count(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(ValueError, match="Expected 2 columns"):
            _identify_columns(df)


# ── Mendeley CSV loading ─────────────────────────────────────────────


class TestLoadMendeleyCSV:
    def test_columns_discovered(self, mendeley_moisture_file: Path) -> None:
        df = load_mendeley_csv(
            mendeley_moisture_file, target_value_name="soil_moisture"
        )
        assert list(df.columns) == ["timestamp", "soil_moisture"]

    def test_date_parsing_monotonic(self, mendeley_moisture_file: Path) -> None:
        df = load_mendeley_csv(mendeley_moisture_file)
        assert df["timestamp"].is_monotonic_increasing

    def test_row_count(self, mendeley_moisture_file: Path) -> None:
        df = load_mendeley_csv(mendeley_moisture_file)
        assert len(df) == 5


# ── Schema validation ────────────────────────────────────────────────


class TestSchemaValidation:
    def test_valid_passes(self, valid_df: pd.DataFrame) -> None:
        validate_schema(valid_df)  # should not raise

    def test_moisture_above_100(self, valid_df: pd.DataFrame) -> None:
        valid_df.loc[0, "soil_moisture"] = 101.0
        with pytest.raises(ValueError, match="soil_moisture"):
            validate_schema(valid_df)

    def test_humidity_below_0(self, valid_df: pd.DataFrame) -> None:
        valid_df.loc[0, "humidity"] = -1.0
        with pytest.raises(ValueError, match="humidity"):
            validate_schema(valid_df)

    def test_missing_column(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.drop(columns=["soil_moisture"])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df)

    def test_irrigation_event_non_binary(self, valid_df: pd.DataFrame) -> None:
        valid_df.loc[0, "irrigation_event"] = 2.0
        with pytest.raises(ValueError, match="irrigation_event"):
            validate_schema(valid_df)


# ── Merge & resample ─────────────────────────────────────────────────


class TestMergeAndResample:
    def test_output_frequency(self) -> None:
        """Output should have exactly 1-hour spacing."""
        n_hours = 24
        ts = pd.date_range("2022-07-12", periods=n_hours * 3, freq="20min")
        df_m = pd.DataFrame({"timestamp": ts, "soil_moisture": 50.0})

        ts_5 = pd.date_range("2022-07-12", periods=n_hours * 12, freq="5min")
        df_v = pd.DataFrame({"timestamp": ts_5, "irrigation_event": 0.0})
        df_f = pd.DataFrame({
            "timestamp": ts_5,
            "flow_l_cumulative": np.arange(len(ts_5), dtype=float),
        })

        ts_h = pd.date_range("2022-07-12", periods=n_hours, freq="1h")
        df_w = pd.DataFrame({
            "timestamp": ts_h,
            "T2M": 25.0,
            "RH2M": 50.0,
            "WS2M": 3.0,
            "ALLSKY_SFC_SW_DWN": 400.0,
        })

        merged = merge_and_resample(df_m, df_v, df_f, df_w, freq="1h")
        diffs = merged["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all()


# ── Provenance ────────────────────────────────────────────────────────


class TestProvenance:
    def test_file_generated(self, tmp_path: Path) -> None:
        config = DataLoaderConfig()
        out = tmp_path / "provenance.yaml"
        generate_provenance(config, out)
        assert out.exists()

    def test_required_keys(self, tmp_path: Path) -> None:
        import yaml as _yaml

        config = DataLoaderConfig()
        out = tmp_path / "provenance.yaml"
        generate_provenance(config, out)
        data = _yaml.safe_load(out.read_text(encoding="utf-8"))
        assert "datasets" in data
        assert "assumptions" in data
        assert "output_conventions" in data
        assert data["output_conventions"]["timestamp_column"]["timezone"] == "UTC"

    def test_utc_documented(self, tmp_path: Path) -> None:
        import yaml as _yaml

        config = DataLoaderConfig()
        out = tmp_path / "provenance.yaml"
        generate_provenance(config, out)
        data = _yaml.safe_load(out.read_text(encoding="utf-8"))
        nasa_ds = data["datasets"][1]
        assert nasa_ds["time_standard"] == "UTC"


# ── Offline mode ──────────────────────────────────────────────────────


class TestOfflineMode:
    def test_use_cache_missing_raises(self, tmp_path: Path) -> None:
        """With use_cache=True and no cached file, fetch should raise."""
        config = NASAPowerConfig()
        with pytest.raises(FileNotFoundError, match="use-cache"):
            fetch_nasa_power(
                config,
                start_date="20220712",
                end_date="20220713",
                cache_dir=tmp_path / "empty_cache",
                use_cache=True,
            )

    def test_use_cache_with_file(self, tmp_path: Path) -> None:
        """With a cached JSON present, fetch should succeed offline."""
        config = NASAPowerConfig(parameters=["T2M"])
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "nasa_power_20220712_20220712.json"
        fake_data = {
            "properties": {
                "parameter": {
                    "T2M": {"2022071200": 25.0, "2022071201": 26.0},
                }
            },
            "header": {"fill_value": -999.0},
        }
        cache_file.write_text(json.dumps(fake_data), encoding="utf-8")

        df = fetch_nasa_power(
            config,
            start_date="20220712",
            end_date="20220712",
            cache_dir=cache_dir,
            use_cache=True,
        )
        assert len(df) == 2
        assert "T2M" in df.columns
