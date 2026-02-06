"""
Tests for SapphirePostprocessingClient.
"""

import warnings

import pytest
import responses
import pandas as pd
from datetime import date

from sapphire_api_client import SapphirePostprocessingClient


class TestPrepareForecastRecords:
    """Tests for prepare_forecast_records static method."""

    def test_basic_preparation(self):
        """Test basic forecast record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "forecast": [100.5, 150.3],
            "lower": [80.0, 120.0],
            "upper": [120.0, 180.0],
        })

        records = SapphirePostprocessingClient.prepare_forecast_records(
            df=df,
            horizon_type="pentad",
            code="12345",
        )

        assert len(records) == 2
        assert records[0]["horizon_type"] == "pentad"
        assert records[0]["code"] == "12345"
        assert records[0]["date"] == "2024-01-01"
        assert records[0]["forecast"] == 100.5
        assert records[0]["lower"] == 80.0
        assert records[0]["upper"] == 120.0

    def test_null_values_handled(self):
        """Test that null values are properly converted to None."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "forecast": [None],
            "lower": [float("nan")],
            "upper": [100.0],
        })

        records = SapphirePostprocessingClient.prepare_forecast_records(
            df=df,
            horizon_type="pentad",
            code="12345",
        )

        assert records[0]["forecast"] is None
        assert records[0]["lower"] is None
        assert records[0]["upper"] == 100.0

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "forecast_date": [date(2024, 1, 1)],
            "predicted_value": [100.0],
            "ci_lower": [80.0],
            "ci_upper": [120.0],
        })

        records = SapphirePostprocessingClient.prepare_forecast_records(
            df=df,
            horizon_type="decade",
            code="54321",
            date_col="forecast_date",
            forecast_col="predicted_value",
            lower_col="ci_lower",
            upper_col="ci_upper",
        )

        assert len(records) == 1
        assert records[0]["date"] == "2024-01-01"
        assert records[0]["forecast"] == 100.0
        assert records[0]["lower"] == 80.0
        assert records[0]["upper"] == 120.0

    def test_without_confidence_bounds(self):
        """Test forecast records without confidence bounds."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "forecast": [100.0],
        })

        records = SapphirePostprocessingClient.prepare_forecast_records(
            df=df,
            horizon_type="month",
            code="12345",
            lower_col=None,
            upper_col=None,
        )

        assert records[0]["forecast"] == 100.0
        assert "lower" not in records[0]
        assert "upper" not in records[0]


class TestPrepareSkillMetricRecords:
    """Tests for prepare_skill_metric_records static method."""

    def test_basic_preparation(self):
        """Test basic skill metric record preparation."""
        df = pd.DataFrame({
            "mae": [10.5],
            "rmse": [15.2],
            "nse": [0.85],
            "kge": [0.78],
            "bias": [-2.3],
            "r2": [0.90],
            "pbias": [-5.0],
        })

        records = SapphirePostprocessingClient.prepare_skill_metric_records(
            df=df,
            horizon_type="pentad",
            code="12345",
            model="linear_regression",
        )

        assert len(records) == 1
        assert records[0]["horizon_type"] == "pentad"
        assert records[0]["code"] == "12345"
        assert records[0]["model"] == "linear_regression"
        assert records[0]["mae"] == 10.5
        assert records[0]["rmse"] == 15.2
        assert records[0]["nse"] == 0.85
        assert records[0]["kge"] == 0.78

    def test_partial_metrics(self):
        """Test with only some metrics present."""
        df = pd.DataFrame({
            "mae": [10.5],
            "nse": [0.85],
            # Other metrics not present
        })

        records = SapphirePostprocessingClient.prepare_skill_metric_records(
            df=df,
            horizon_type="decade",
            code="12345",
            model="ml_model",
        )

        assert records[0]["mae"] == 10.5
        assert records[0]["nse"] == 0.85
        assert "rmse" not in records[0]
        assert "kge" not in records[0]

    def test_null_metric_values(self):
        """Test handling of null metric values."""
        df = pd.DataFrame({
            "mae": [10.5],
            "rmse": [None],
            "nse": [float("nan")],
        })

        records = SapphirePostprocessingClient.prepare_skill_metric_records(
            df=df,
            horizon_type="pentad",
            code="12345",
            model="test_model",
        )

        assert records[0]["mae"] == 10.5
        assert records[0]["rmse"] is None
        assert records[0]["nse"] is None


class TestPrepareForecastRecordsValidation:
    """Tests for error handling in prepare_forecast_records."""

    def test_missing_date_column(self):
        """Test that missing date column raises ValueError."""
        df = pd.DataFrame({"forecast": [100.0]})

        with pytest.raises(ValueError, match="missing required columns.*fecha"):
            SapphirePostprocessingClient.prepare_forecast_records(
                df=df, horizon_type="pentad", code="12345", date_col="fecha"
            )


class TestPostprocessingClientAPI:
    """Tests for PostprocessingClient API calls."""

    def setup_method(self):
        """Set up test client."""
        self.client = SapphirePostprocessingClient(
            base_url="http://localhost:8000",
            max_retries=1,
        )

    @responses.activate
    def test_read_forecasts(self):
        """Test reading forecast data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[
                {"id": 1, "code": "12345", "date": "2024-01-01", "forecast": 100.0}
            ],
            status=200,
        )

        df = self.client.read_forecasts(code="12345")

        assert len(df) == 1
        assert df.iloc[0]["code"] == "12345"
        assert df.iloc[0]["forecast"] == 100.0

    @responses.activate
    def test_read_forecasts_with_filters(self):
        """Test reading forecasts with all filters."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[],
            status=200,
        )

        df = self.client.read_forecasts(
            horizon="pentad",
            code="12345",
            start_date="2024-01-01",
            end_date="2024-01-31",
            skip=0,
            limit=50,
        )

        # Check that params were sent correctly
        assert "horizon=pentad" in responses.calls[0].request.url
        assert "code=12345" in responses.calls[0].request.url

    @responses.activate
    def test_write_forecasts(self):
        """Test writing forecast records."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[{"id": 1}],
            status=201,
        )

        records = [{"horizon_type": "pentad", "code": "12345", "date": "2024-01-01", "forecast": 100.0}]
        count = self.client.write_forecasts(records)

        assert count == 1

    @responses.activate
    def test_read_lr_forecasts(self):
        """Test reading linear regression forecast data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/lr-forecast/",
            json=[{"id": 1, "code": "12345", "forecast": 95.0}],
            status=200,
        )

        df = self.client.read_lr_forecasts(horizon="pentad", code="12345")
        assert len(df) == 1
        assert df.iloc[0]["forecast"] == 95.0

    @responses.activate
    def test_write_lr_forecasts(self):
        """Test writing linear regression forecast records."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/lr-forecast/",
            json=[{"id": 1}, {"id": 2}],
            status=201,
        )

        records = [
            {"horizon_type": "pentad", "code": "12345", "forecast": 95.0},
            {"horizon_type": "pentad", "code": "12345", "forecast": 100.0},
        ]
        count = self.client.write_lr_forecasts(records)

        assert count == 2

    @responses.activate
    def test_read_skill_metrics(self):
        """Test reading skill metrics."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/skill-metric/",
            json=[{"id": 1, "code": "12345", "model": "lr", "nse": 0.85}],
            status=200,
        )

        df = self.client.read_skill_metrics(horizon="pentad", code="12345", model="lr")
        assert len(df) == 1
        assert df.iloc[0]["nse"] == 0.85

    @responses.activate
    def test_read_skill_metrics_with_model_filter(self):
        """Test reading skill metrics with model filter."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/skill-metric/",
            json=[],
            status=200,
        )

        df = self.client.read_skill_metrics(
            horizon="decade",
            code="12345",
            model="conceptual_model",
        )

        assert "model=conceptual_model" in responses.calls[0].request.url

    @responses.activate
    def test_write_skill_metrics(self):
        """Test writing skill metric records."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/skill-metric/",
            json=[{"id": 1}],
            status=201,
        )

        records = [{"horizon_type": "pentad", "code": "12345", "model": "lr", "nse": 0.85}]
        count = self.client.write_skill_metrics(records)

        assert count == 1

    @responses.activate
    def test_empty_response_returns_empty_dataframe(self):
        """Test that empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[],
            status=200,
        )

        df = self.client.read_forecasts()
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    @responses.activate
    def test_health_check_inherited(self):
        """Test that health check is inherited from base client."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/health",
            json={"status": "healthy"},
            status=200,
        )

        assert self.client.health_check() is True


class TestPostprocessingInputValidation:
    """Tests for input validation in postprocessing read methods."""

    def setup_method(self):
        self.client = SapphirePostprocessingClient(
            base_url="http://localhost:8000", max_retries=1
        )

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="Invalid horizon 'weekly'"):
            self.client.read_forecasts(horizon="weekly")

    def test_invalid_horizon_skill_metrics(self):
        with pytest.raises(ValueError, match="Invalid horizon"):
            self.client.read_skill_metrics(horizon="biweekly")

    def test_negative_skip_raises(self):
        with pytest.raises(ValueError, match="skip must be non-negative"):
            self.client.read_forecasts(skip=-1)

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            self.client.read_forecasts(limit=0)

    def test_missing_metric_columns_warns(self):
        """Test warning when no metric columns found."""
        df = pd.DataFrame({"unrelated_col": [1.0]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            records = SapphirePostprocessingClient.prepare_skill_metric_records(
                df=df, horizon_type="pentad", code="12345", model="LR"
            )
            assert len(w) == 1
            assert "No metric columns found" in str(w[0].message)
        assert len(records) == 1
