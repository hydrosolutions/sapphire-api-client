"""
Tests for SapphireShortTermForecastClient.
"""

import pytest
import responses
import pandas as pd
from datetime import date

from sapphire_api_client import SapphireShortTermForecastClient


class TestPrepareShortTermForecastRecords:
    """Tests for prepare_short_term_forecast_records static method."""

    def test_basic_preparation(self):
        """Test basic forecast record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "forecast": [100.5, 150.3],
            "lower": [80.0, 120.0],
            "upper": [120.0, 180.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
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

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df,
            horizon_type="pentad",
            code="12345",
        )

        assert records[0]["forecast"] is None
        assert records[0]["lower"] is None
        assert records[0]["upper"] == 100.0

    @pytest.mark.parametrize(
        "horizon_type",
        ["day", "pentad", "decade", "month", "quarter", "season", "year"],
    )
    def test_all_server_horizons_accepted(self, horizon_type):
        """Every server HorizonType value (incl. quarter) flows through."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "forecast": [100.5],
            "lower": [80.0],
            "upper": [120.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df,
            horizon_type=horizon_type,
            code="12345",
        )

        assert records[0]["horizon_type"] == horizon_type

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "forecast_date": [date(2024, 1, 1)],
            "predicted_value": [100.0],
            "ci_lower": [80.0],
            "ci_upper": [120.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
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

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df,
            horizon_type="month",
            code="12345",
            lower_col=None,
            upper_col=None,
        )

        assert records[0]["forecast"] == 100.0
        assert "lower" not in records[0]
        assert "upper" not in records[0]

    def test_missing_date_column(self):
        """Test that missing date column raises ValueError."""
        df = pd.DataFrame({"forecast": [100.0]})

        with pytest.raises(ValueError, match="missing required columns.*fecha"):
            SapphireShortTermForecastClient.prepare_short_term_forecast_records(
                df=df, horizon_type="pentad", code="12345", date_col="fecha"
            )


class TestPrepareShortTermForecastRecordsEdgeCases:
    """Edge case tests for prepare_short_term_forecast_records."""

    def test_empty_dataframe_returns_empty_list(self):
        """Empty DataFrame produces no records."""
        df = pd.DataFrame(columns=["date", "forecast", "lower", "upper"])

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="pentad", code="12345",
        )

        assert records == []

    def test_all_nan_forecast_values(self):
        """All-NaN forecast and bound columns convert to None."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "forecast": [float("nan"), float("nan")],
            "lower": [float("nan"), float("nan")],
            "upper": [float("nan"), float("nan")],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="pentad", code="12345",
        )

        assert len(records) == 2
        for r in records:
            assert r["forecast"] is None
            assert r["lower"] is None
            assert r["upper"] is None

    def test_zero_forecast_value_preserved(self):
        """Zero is a valid forecast value, not treated as missing."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "forecast": [0.0],
            "lower": [0.0],
            "upper": [0.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="day", code="12345",
        )

        assert records[0]["forecast"] == 0.0
        assert records[0]["lower"] == 0.0
        assert records[0]["upper"] == 0.0

    def test_extreme_values(self):
        """Very small and very large values pass through unchanged."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "forecast": [0.001, 99999.99],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="pentad", code="12345",
            lower_col=None, upper_col=None,
        )

        assert records[0]["forecast"] == 0.001
        assert records[1]["forecast"] == 99999.99

    def test_multi_entity_same_code(self):
        """Multiple rows all receive the same code parameter."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "forecast": [10.0, 20.0, 30.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="pentad", code="99999",
            lower_col=None, upper_col=None,
        )

        assert len(records) == 3
        assert all(r["code"] == "99999" for r in records)
        assert [r["forecast"] for r in records] == [10.0, 20.0, 30.0]

    def test_row_order_preserved(self):
        """Output record order matches input DataFrame row order."""
        dates = [date(2024, 3, 15), date(2024, 1, 1), date(2024, 6, 30)]
        df = pd.DataFrame({
            "date": dates,
            "forecast": [300.0, 100.0, 600.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="day", code="12345",
            lower_col=None, upper_col=None,
        )

        assert [r["date"] for r in records] == ["2024-03-15", "2024-01-01", "2024-06-30"]
        assert [r["forecast"] for r in records] == [300.0, 100.0, 600.0]

    def test_year_transition_dates(self):
        """Dec 31 and Jan 1 dates serialize correctly."""
        df = pd.DataFrame({
            "date": [date(2024, 12, 31), date(2025, 1, 1)],
            "forecast": [50.0, 55.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="day", code="12345",
            lower_col=None, upper_col=None,
        )

        assert records[0]["date"] == "2024-12-31"
        assert records[1]["date"] == "2025-01-01"

    def test_leap_year_date(self):
        """Feb 29 on a leap year serializes correctly."""
        df = pd.DataFrame({
            "date": [date(2024, 2, 29)],
            "forecast": [42.0],
        })

        records = SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df, horizon_type="day", code="12345",
            lower_col=None, upper_col=None,
        )

        assert records[0]["date"] == "2024-02-29"


class TestShortTermForecastClientAPI:
    """Tests for ShortTermForecastClient API calls."""

    def setup_method(self):
        """Set up test client."""
        self.client = SapphireShortTermForecastClient(
            base_url="http://localhost:8000",
            max_retries=1,
        )

    @responses.activate
    def test_read_short_term_forecasts(self):
        """Test reading short-term forecast data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[
                {"id": 1, "code": "12345", "date": "2024-01-01", "forecast": 100.0}
            ],
            status=200,
        )

        df = self.client.read_short_term_forecasts(code="12345")

        assert len(df) == 1
        assert df.iloc[0]["code"] == "12345"
        assert df.iloc[0]["forecast"] == 100.0

    @responses.activate
    def test_read_short_term_forecasts_with_filters(self):
        """Test reading short-term forecasts with all filters."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[],
            status=200,
        )

        df = self.client.read_short_term_forecasts(
            horizon="pentad",
            code="12345",
            start_date="2024-01-01",
            end_date="2024-01-31",
            skip=0,
            limit=50,
        )

        assert "horizon=pentad" in responses.calls[0].request.url
        assert "code=12345" in responses.calls[0].request.url

    @responses.activate
    def test_read_short_term_forecasts_target_params(self):
        """Test that target, start_target, end_target reach the URL."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[],
            status=200,
        )

        self.client.read_short_term_forecasts(
            target="2024-06-15",
            start_target="2024-06-01",
            end_target="2024-06-30",
        )

        url = responses.calls[0].request.url
        assert "target=2024-06-15" in url
        assert "start_target=2024-06-01" in url
        assert "end_target=2024-06-30" in url

    @responses.activate
    def test_write_short_term_forecasts(self):
        """Test writing short-term forecast records."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[{"id": 1}],
            status=201,
        )

        records = [{"horizon_type": "pentad", "code": "12345", "date": "2024-01-01", "forecast": 100.0}]
        count = self.client.write_short_term_forecasts(records)

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
    def test_empty_response_returns_empty_dataframe(self):
        """Test that empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[],
            status=200,
        )

        df = self.client.read_short_term_forecasts()
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


class TestShortTermForecastInputValidation:
    """Tests for input validation in short-term forecast read methods."""

    def setup_method(self):
        self.client = SapphireShortTermForecastClient(
            base_url="http://localhost:8000", max_retries=1
        )

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="Invalid horizon 'weekly'"):
            self.client.read_short_term_forecasts(horizon="weekly")

    def test_negative_skip_raises(self):
        with pytest.raises(ValueError, match="skip must be non-negative"):
            self.client.read_short_term_forecasts(skip=-1)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Invalid model 'INVALID'"):
            self.client.read_short_term_forecasts(model="INVALID")

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            self.client.read_short_term_forecasts(limit=0)

    def test_lr_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="Invalid horizon 'weekly'"):
            self.client.read_lr_forecasts(horizon="weekly")

    def test_lr_negative_skip_raises(self):
        with pytest.raises(ValueError, match="skip must be non-negative"):
            self.client.read_lr_forecasts(skip=-1)

    def test_lr_zero_limit_raises(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            self.client.read_lr_forecasts(limit=0)
