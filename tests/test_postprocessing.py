"""
Tests for the SapphirePostprocessingClient facade.

Covers:
- Inheritance structure (has all methods from both forecast families)
- Deprecated alias warnings
- Skill metrics (live on the facade via the base class)
"""

import warnings

import pytest
import responses
import pandas as pd
from datetime import date

from sapphire_api_client import SapphirePostprocessingClient
from sapphire_api_client.long_term import SapphireLongTermForecastClient
from sapphire_api_client.postprocessing_base import SapphirePostprocessingBase
from sapphire_api_client.short_term import SapphireShortTermForecastClient


class TestInheritance:
    """Verify that the facade inherits from both forecast families."""

    def test_is_subclass_of_short_term(self):
        assert issubclass(SapphirePostprocessingClient, SapphireShortTermForecastClient)

    def test_is_subclass_of_long_term(self):
        assert issubclass(SapphirePostprocessingClient, SapphireLongTermForecastClient)

    def test_is_subclass_of_base(self):
        assert issubclass(SapphirePostprocessingClient, SapphirePostprocessingBase)

    def test_has_short_term_methods(self):
        client = SapphirePostprocessingClient(base_url="http://localhost:8000")
        assert hasattr(client, "read_short_term_forecasts")
        assert hasattr(client, "write_short_term_forecasts")
        assert hasattr(client, "prepare_short_term_forecast_records")
        assert hasattr(client, "read_lr_forecasts")
        assert hasattr(client, "write_lr_forecasts")

    def test_has_long_term_methods(self):
        client = SapphirePostprocessingClient(base_url="http://localhost:8000")
        assert hasattr(client, "read_long_term_forecasts")
        assert hasattr(client, "write_long_term_forecasts")
        assert hasattr(client, "prepare_long_term_forecast_records")

    def test_has_skill_metric_methods(self):
        client = SapphirePostprocessingClient(base_url="http://localhost:8000")
        assert hasattr(client, "read_skill_metrics")
        assert hasattr(client, "write_skill_metrics")
        assert hasattr(client, "prepare_skill_metric_records")

    def test_service_prefix(self):
        client = SapphirePostprocessingClient(base_url="http://localhost:8000")
        assert client.SERVICE_PREFIX == "/api/postprocessing"


class TestDeprecatedAliases:
    """Verify that old method names emit DeprecationWarning and delegate."""

    def setup_method(self):
        self.client = SapphirePostprocessingClient(
            base_url="http://localhost:8000", max_retries=1
        )

    @responses.activate
    def test_read_forecasts_warns(self):
        """read_forecasts() emits DeprecationWarning and returns data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[{"id": 1, "code": "12345", "forecast": 100.0}],
            status=200,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = self.client.read_forecasts(code="12345")

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "read_short_term_forecasts" in str(w[0].message)
        assert len(df) == 1
        assert df.iloc[0]["forecast"] == 100.0

    @responses.activate
    def test_write_forecasts_warns(self):
        """write_forecasts() emits DeprecationWarning and writes data."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/forecast/",
            json=[{"id": 1}],
            status=201,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            count = self.client.write_forecasts([{"code": "12345"}])

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "write_short_term_forecasts" in str(w[0].message)
        assert count == 1

    def test_prepare_forecast_records_warns(self):
        """prepare_forecast_records() emits DeprecationWarning."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "forecast": [100.0],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            records = SapphirePostprocessingClient.prepare_forecast_records(
                df=df, horizon_type="pentad", code="12345",
            )

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "prepare_short_term_forecast_records" in str(w[0].message)
        assert len(records) == 1
        assert records[0]["forecast"] == 100.0

    @responses.activate
    def test_read_long_forecasts_warns(self):
        """read_long_forecasts() emits DeprecationWarning and returns data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[{"id": 1, "code": "15013", "q": 123.45}],
            status=200,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = self.client.read_long_forecasts(code="15013")

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "read_long_term_forecasts" in str(w[0].message)
        assert len(df) == 1
        assert df.iloc[0]["q"] == 123.45

    @responses.activate
    def test_write_long_forecasts_warns(self):
        """write_long_forecasts() emits DeprecationWarning and writes data."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/long-forecast/",
            json=[{"id": 1}],
            status=201,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            count = self.client.write_long_forecasts([{"code": "15013"}])

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "write_long_term_forecasts" in str(w[0].message)
        assert count == 1

    def test_prepare_long_forecast_records_warns(self):
        """prepare_long_forecast_records() emits DeprecationWarning."""
        df = pd.DataFrame({
            "code": ["15013"],
            "date": [date(2024, 6, 15)],
            "valid_from": [date(2024, 7, 1)],
            "valid_to": [date(2024, 7, 31)],
        })

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            records = SapphirePostprocessingClient.prepare_long_forecast_records(
                df=df, horizon_type="month", horizon_value=7, model_type="GBT",
            )

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "prepare_long_term_forecast_records" in str(w[0].message)
        assert len(records) == 1
        assert records[0]["code"] == "15013"


class TestSkillMetricsOnFacade:
    """Skill metric methods accessed through the facade."""

    def setup_method(self):
        self.client = SapphirePostprocessingClient(
            base_url="http://localhost:8000", max_retries=1
        )

    @responses.activate
    def test_read_skill_metrics(self):
        """Test reading skill metrics via facade."""
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

        self.client.read_skill_metrics(
            horizon="decade",
            code="12345",
            model="conceptual_model",
        )

        assert "model=conceptual_model" in responses.calls[0].request.url

    @responses.activate
    def test_write_skill_metrics(self):
        """Test writing skill metric records via facade."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/postprocessing/skill-metric/",
            json=[{"id": 1}],
            status=201,
        )

        records = [{"horizon_type": "pentad", "code": "12345", "model": "lr", "nse": 0.85}]
        count = self.client.write_skill_metrics(records)

        assert count == 1

    def test_prepare_skill_metric_records_basic(self):
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

    def test_prepare_skill_metric_records_partial(self):
        """Test with only some metrics present."""
        df = pd.DataFrame({
            "mae": [10.5],
            "nse": [0.85],
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

    def test_prepare_skill_metric_records_null_values(self):
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


class TestPostprocessingInputValidation:
    """Tests for input validation on the facade."""

    def setup_method(self):
        self.client = SapphirePostprocessingClient(
            base_url="http://localhost:8000", max_retries=1
        )

    def test_invalid_horizon_skill_metrics(self):
        with pytest.raises(ValueError, match="Invalid horizon"):
            self.client.read_skill_metrics(horizon="biweekly")

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
