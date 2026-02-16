"""
Tests for SapphireLongTermForecastClient.
"""

import pytest
import responses
import pandas as pd
from datetime import date

from sapphire_api_client import SapphireLongTermForecastClient


class TestPrepareLongTermForecastRecords:
    """Tests for prepare_long_term_forecast_records static method."""

    def test_basic_preparation(self):
        """Test full record preparation with all quantile columns."""
        df = pd.DataFrame({
            "code": ["15013"],
            "date": [date(2024, 6, 15)],
            "valid_from": [date(2024, 7, 1)],
            "valid_to": [date(2024, 7, 31)],
            "flag": [0],
            "composition": ["GBT+LR"],
            "q": [123.45],
            "q_obs": [120.0],
            "q_xgb": [125.0],
            "q_lgbm": [124.0],
            "q_catboost": [123.0],
            "q_loc": [122.0],
            "q05": [100.0],
            "q10": [110.0],
            "q25": [115.0],
            "q50": [123.0],
            "q75": [130.0],
            "q90": [135.0],
            "q95": [140.0],
        })

        records = SapphireLongTermForecastClient.prepare_long_term_forecast_records(
            df=df,
            horizon_type="month",
            horizon_value=7,
            model_type="GBT",
        )

        assert len(records) == 1
        r = records[0]
        assert r["horizon_type"] == "month"
        assert r["horizon_value"] == 7
        assert r["code"] == "15013"
        assert r["date"] == "2024-06-15"
        assert r["model_type"] == "GBT"
        assert r["valid_from"] == "2024-07-01"
        assert r["valid_to"] == "2024-07-31"
        assert r["flag"] == 0
        assert r["composition"] == "GBT+LR"
        assert r["q"] == 123.45
        assert r["q_obs"] == 120.0
        assert r["q05"] == 100.0
        assert r["q95"] == 140.0

    def test_nan_to_none_conversion(self):
        """Test that NaN values are converted to None."""
        df = pd.DataFrame({
            "code": ["15013"],
            "date": [date(2024, 6, 15)],
            "valid_from": [date(2024, 7, 1)],
            "valid_to": [date(2024, 7, 31)],
            "flag": [None],
            "q": [float("nan")],
            "q_obs": [120.0],
            "q50": [None],
        })

        records = SapphireLongTermForecastClient.prepare_long_term_forecast_records(
            df=df,
            horizon_type="month",
            horizon_value=7,
            model_type="GBT",
        )

        r = records[0]
        assert r["flag"] is None
        assert r["q"] is None
        assert r["q_obs"] == 120.0
        assert r["q50"] is None

    def test_missing_optional_cols(self):
        """Test with only required columns present."""
        df = pd.DataFrame({
            "code": ["15013"],
            "date": [date(2024, 6, 15)],
            "valid_from": [date(2024, 7, 1)],
            "valid_to": [date(2024, 7, 31)],
        })

        records = SapphireLongTermForecastClient.prepare_long_term_forecast_records(
            df=df,
            horizon_type="season",
            horizon_value=1,
            model_type="LR_Base",
        )

        assert len(records) == 1
        r = records[0]
        assert r["horizon_type"] == "season"
        assert r["code"] == "15013"
        assert r["model_type"] == "LR_Base"
        # Optional fields should not be present
        assert "flag" not in r
        assert "q" not in r
        assert "q50" not in r

    def test_missing_required_col_raises(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({
            "code": ["15013"],
            "date": [date(2024, 6, 15)],
            # missing valid_from and valid_to
        })

        with pytest.raises(ValueError, match="missing required columns.*valid_from"):
            SapphireLongTermForecastClient.prepare_long_term_forecast_records(
                df=df,
                horizon_type="month",
                horizon_value=7,
                model_type="GBT",
            )


class TestLongTermForecastClientAPI:
    """Tests for LongTermForecastClient API calls."""

    def setup_method(self):
        """Set up test client."""
        self.client = SapphireLongTermForecastClient(
            base_url="http://localhost:8000",
            max_retries=1,
        )

    @responses.activate
    def test_read_long_term_forecasts_returns_dataframe(self):
        """Test that read_long_term_forecasts returns a DataFrame with records."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[
                {
                    "id": 1,
                    "horizon_type": "month",
                    "horizon_value": 7,
                    "code": "15013",
                    "date": "2024-06-15",
                    "model_type": "GBT",
                    "valid_from": "2024-07-01",
                    "valid_to": "2024-07-31",
                    "q": 123.45,
                    "q50": 123.0,
                }
            ],
            status=200,
        )

        df = self.client.read_long_term_forecasts(code="15013")

        assert len(df) == 1
        assert df.iloc[0]["code"] == "15013"
        assert df.iloc[0]["model_type"] == "GBT"
        assert df.iloc[0]["q"] == 123.45
        assert df.iloc[0]["horizon_value"] == 7

    @responses.activate
    def test_read_long_term_forecasts_empty(self):
        """Test that empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[],
            status=200,
        )

        df = self.client.read_long_term_forecasts()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @responses.activate
    def test_read_long_term_forecasts_builds_params(self):
        """Test that all query params are passed correctly."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[],
            status=200,
        )

        self.client.read_long_term_forecasts(
            horizon_type="month",
            horizon_value=0,
            code="15013",
            model="GBT",
            start_date="2024-01-01",
            end_date="2024-12-31",
            valid_from="2024-07-01",
            valid_to="2024-07-31",
            skip=10,
            limit=50,
        )

        url = responses.calls[0].request.url
        assert "horizon_type=month" in url
        # horizon_value=0 must be included (is not None check)
        assert "horizon_value=0" in url
        assert "code=15013" in url
        assert "model=GBT" in url
        assert "start_date=2024-01-01" in url
        assert "end_date=2024-12-31" in url
        assert "valid_from=2024-07-01" in url
        assert "valid_to=2024-07-31" in url
        assert "skip=10" in url
        assert "limit=50" in url
        # Must NOT use short-forecast param names
        assert "horizon=" not in url or "horizon_type" in url

    @responses.activate
    def test_write_long_term_forecasts_calls_post_batched(self):
        """Test that write_long_term_forecasts posts to /long-forecast/."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[{"id": 1}, {"id": 2}],
            status=201,
        )

        records = [
            {
                "horizon_type": "month",
                "horizon_value": 7,
                "code": "15013",
                "date": "2024-06-15",
                "model_type": "GBT",
                "valid_from": "2024-07-01",
                "valid_to": "2024-07-31",
                "q": 123.45,
            },
            {
                "horizon_type": "month",
                "horizon_value": 7,
                "code": "15014",
                "date": "2024-06-15",
                "model_type": "GBT",
                "valid_from": "2024-07-01",
                "valid_to": "2024-07-31",
                "q": 200.0,
            },
        ]
        count = self.client.write_long_term_forecasts(records)

        assert count == 2


class TestLongTermForecastInputValidation:
    """Tests for input validation in long-term forecast read methods."""

    def setup_method(self):
        self.client = SapphireLongTermForecastClient(
            base_url="http://localhost:8000", max_retries=1
        )

    def test_invalid_horizon_type_raises(self):
        """Test that invalid horizon_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid horizon_type 'weekly'"):
            self.client.read_long_term_forecasts(horizon_type="weekly")

    def test_invalid_model_raises(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model 'InvalidModel'"):
            self.client.read_long_term_forecasts(model="InvalidModel")

    def test_negative_skip_raises(self):
        """Test that negative skip raises ValueError."""
        with pytest.raises(ValueError, match="skip must be non-negative"):
            self.client.read_long_term_forecasts(skip=-1)

    def test_zero_limit_raises(self):
        """Test that zero limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be positive"):
            self.client.read_long_term_forecasts(limit=0)
