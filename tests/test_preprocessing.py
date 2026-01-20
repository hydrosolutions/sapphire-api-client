"""
Tests for SapphirePreprocessingClient.
"""

import pytest
import responses
import pandas as pd
from datetime import date

from sapphire_api_client import SapphirePreprocessingClient


class TestPrepareRunoffRecords:
    """Tests for prepare_runoff_records static method."""

    def test_basic_preparation(self):
        """Test basic runoff record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "discharge": [100.5, 150.3],
            "predictor": [50.0, 60.0],
            "horizon_value": [1, 1],
            "horizon_in_year": [1, 2],
        })

        records = SapphirePreprocessingClient.prepare_runoff_records(
            df=df,
            horizon_type="day",
            code="12345",
        )

        assert len(records) == 2
        assert records[0]["horizon_type"] == "day"
        assert records[0]["code"] == "12345"
        assert records[0]["date"] == "2024-01-01"
        assert records[0]["discharge"] == 100.5
        assert records[0]["predictor"] == 50.0
        assert records[0]["horizon_value"] == 1
        assert records[0]["horizon_in_year"] == 1

    def test_null_values_handled(self):
        """Test that null values are properly converted to None."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "discharge": [None],
            "predictor": [float("nan")],
            "horizon_value": [1],
            "horizon_in_year": [1],
        })

        records = SapphirePreprocessingClient.prepare_runoff_records(
            df=df,
            horizon_type="day",
            code="12345",
        )

        assert records[0]["discharge"] is None
        assert records[0]["predictor"] is None

    def test_custom_column_names(self):
        """Test with custom column names."""
        df = pd.DataFrame({
            "my_date": [date(2024, 1, 1)],
            "q_value": [100.0],
            "hv": [1],
            "hiy": [1],
        })

        records = SapphirePreprocessingClient.prepare_runoff_records(
            df=df,
            horizon_type="pentad",
            code="54321",
            date_col="my_date",
            discharge_col="q_value",
            predictor_col=None,
            horizon_value_col="hv",
            horizon_in_year_col="hiy",
        )

        assert len(records) == 1
        assert records[0]["date"] == "2024-01-01"
        assert records[0]["discharge"] == 100.0
        assert "predictor" not in records[0]


class TestPrepareHydrographRecords:
    """Tests for prepare_hydrograph_records static method."""

    def test_basic_preparation(self):
        """Test basic hydrograph record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "day_of_year": [1],
            "horizon_value": [1],
            "horizon_in_year": [1],
            "mean": [100.0],
            "std": [10.0],
            "min": [80.0],
            "max": [120.0],
            "q50": [99.0],
        })

        records = SapphirePreprocessingClient.prepare_hydrograph_records(
            df=df,
            horizon_type="day",
            code="12345",
        )

        assert len(records) == 1
        assert records[0]["mean"] == 100.0
        assert records[0]["std"] == 10.0
        assert records[0]["q50"] == 99.0

    def test_missing_stat_columns(self):
        """Test handling when some stat columns are missing."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "day_of_year": [1],
            "horizon_value": [1],
            "horizon_in_year": [1],
            "mean": [100.0],
            # Other stat columns missing
        })

        records = SapphirePreprocessingClient.prepare_hydrograph_records(
            df=df,
            horizon_type="day",
            code="12345",
        )

        assert records[0]["mean"] == 100.0
        # Missing columns should not be in record
        assert "std" not in records[0]


class TestPrepareMeteoRecords:
    """Tests for prepare_meteo_records static method."""

    def test_temperature_preparation(self):
        """Test temperature record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "value": [15.5],
            "norm": [12.0],
            "day_of_year": [1],
        })

        records = SapphirePreprocessingClient.prepare_meteo_records(
            df=df,
            meteo_type="T",
            code="12345",
        )

        assert len(records) == 1
        assert records[0]["meteo_type"] == "T"
        assert records[0]["value"] == 15.5
        assert records[0]["norm"] == 12.0

    def test_precipitation_preparation(self):
        """Test precipitation record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 1)],
            "value": [5.2],
            "day_of_year": [1],
        })

        records = SapphirePreprocessingClient.prepare_meteo_records(
            df=df,
            meteo_type="P",
            code="12345",
            norm_col=None,
        )

        assert records[0]["meteo_type"] == "P"
        assert records[0]["value"] == 5.2
        assert "norm" not in records[0]


class TestPrepareSnowRecords:
    """Tests for prepare_snow_records static method."""

    def test_basic_snow_preparation(self):
        """Test basic snow record preparation."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 15)],
            "value": [50.0],
            "norm": [45.0],
        })

        records = SapphirePreprocessingClient.prepare_snow_records(
            df=df,
            snow_type="HS",
            code="12345",
        )

        assert len(records) == 1
        assert records[0]["snow_type"] == "HS"
        assert records[0]["value"] == 50.0
        assert records[0]["norm"] == 45.0

    def test_zone_values(self):
        """Test snow records with zone-specific values."""
        df = pd.DataFrame({
            "date": [date(2024, 1, 15)],
            "value": [50.0],
            "value1": [40.0],
            "value2": [50.0],
            "value3": [60.0],
        })

        records = SapphirePreprocessingClient.prepare_snow_records(
            df=df,
            snow_type="SWE",
            code="12345",
        )

        assert records[0]["value1"] == 40.0
        assert records[0]["value2"] == 50.0
        assert records[0]["value3"] == 60.0


class TestPreprocessingClientAPI:
    """Tests for PreprocessingClient API calls."""

    def setup_method(self):
        """Set up test client."""
        self.client = SapphirePreprocessingClient(
            base_url="http://localhost:8000",
            max_retries=1,
        )

    @responses.activate
    def test_read_runoff(self):
        """Test reading runoff data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/runoff/",
            json=[
                {"id": 1, "code": "12345", "date": "2024-01-01", "discharge": 100.0}
            ],
            status=200,
        )

        df = self.client.read_runoff(code="12345")

        assert len(df) == 1
        assert df.iloc[0]["code"] == "12345"

    @responses.activate
    def test_read_runoff_with_filters(self):
        """Test reading runoff with all filters."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/runoff/",
            json=[],
            status=200,
        )

        df = self.client.read_runoff(
            horizon="day",
            code="12345",
            start_date="2024-01-01",
            end_date="2024-01-31",
            skip=0,
            limit=50,
        )

        # Check that params were sent correctly
        assert "horizon=day" in responses.calls[0].request.url
        assert "code=12345" in responses.calls[0].request.url

    @responses.activate
    def test_write_runoff(self):
        """Test writing runoff records."""
        responses.add(
            responses.POST,
            "http://localhost:8000/api/preprocessing/runoff/",
            json=[{"id": 1}],
            status=201,
        )

        records = [{"horizon_type": "day", "code": "12345", "date": "2024-01-01"}]
        count = self.client.write_runoff(records)

        assert count == 1

    @responses.activate
    def test_read_hydrograph(self):
        """Test reading hydrograph data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/hydrograph/",
            json=[{"id": 1, "mean": 100.0}],
            status=200,
        )

        df = self.client.read_hydrograph(horizon="day", code="12345")
        assert len(df) == 1

    @responses.activate
    def test_read_meteo(self):
        """Test reading meteo data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/meteo/",
            json=[{"id": 1, "meteo_type": "T", "value": 15.0}],
            status=200,
        )

        df = self.client.read_meteo(meteo_type="T", code="12345")
        assert len(df) == 1

    @responses.activate
    def test_read_snow(self):
        """Test reading snow data."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/snow/",
            json=[{"id": 1, "snow_type": "HS", "value": 50.0}],
            status=200,
        )

        df = self.client.read_snow(snow_type="HS", code="12345")
        assert len(df) == 1

    @responses.activate
    def test_empty_response_returns_empty_dataframe(self):
        """Test that empty API response returns empty DataFrame."""
        responses.add(
            responses.GET,
            "http://localhost:8000/api/preprocessing/runoff/",
            json=[],
            status=200,
        )

        df = self.client.read_runoff()
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)
