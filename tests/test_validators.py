"""
Tests for the validators module.
"""

import warnings

import pytest
import pandas as pd

from sapphire_api_client.validators import (
    VALID_HORIZONS,
    VALID_METEO_TYPES,
    VALID_SNOW_TYPES,
    VALID_FORECAST_MODELS,
    validate_base_url,
    validate_positive_int,
    validate_non_negative_int,
    warn_http_with_token,
    validate_enum_param,
    truncate_response_text,
    safe_int_conversion,
)


# ==================== Constants ====================


class TestConstants:
    """Tests for exported constant sets."""

    def test_valid_horizons(self):
        assert VALID_HORIZONS == {"day", "pentad", "decade", "month", "season", "year"}

    def test_valid_meteo_types(self):
        assert VALID_METEO_TYPES == {"T", "P"}

    def test_valid_snow_types(self):
        assert VALID_SNOW_TYPES == {"HS", "ROF", "SWE"}

    def test_valid_forecast_models(self):
        assert VALID_FORECAST_MODELS == {"TFT", "TiDE", "TSMixer", "LR", "EM", "NE"}


# ==================== validate_base_url ====================


class TestValidateBaseUrl:
    """Tests for validate_base_url."""

    def test_valid_https_url(self):
        validate_base_url("https://api.example.com")  # should not raise

    def test_valid_http_url(self):
        validate_base_url("http://localhost:8000")  # should not raise

    def test_invalid_scheme_ftp(self):
        with pytest.raises(ValueError, match="Invalid URL scheme 'ftp'"):
            validate_base_url("ftp://files.example.com")

    def test_empty_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_base_url("://no-scheme.com")

    def test_missing_host(self):
        with pytest.raises(ValueError, match="missing host"):
            validate_base_url("http://")

    def test_no_scheme_at_all(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_base_url("just-a-string")


# ==================== validate_positive_int ====================


class TestValidatePositiveInt:
    """Tests for validate_positive_int."""

    def test_valid_positive(self):
        validate_positive_int(1, "test")  # should not raise
        validate_positive_int(100, "test")  # should not raise

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_int(0, "batch_size")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive, got -5"):
            validate_positive_int(-5, "max_retries")


# ==================== validate_non_negative_int ====================


class TestValidateNonNegativeInt:
    """Tests for validate_non_negative_int."""

    def test_valid_zero(self):
        validate_non_negative_int(0, "skip")  # should not raise

    def test_valid_positive(self):
        validate_non_negative_int(10, "skip")  # should not raise

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be non-negative, got -1"):
            validate_non_negative_int(-1, "skip")


# ==================== warn_http_with_token ====================


class TestWarnHttpWithToken:
    """Tests for warn_http_with_token."""

    def test_http_with_token_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_http_with_token("http://api.example.com", has_token=True)
            assert len(w) == 1
            assert "plain HTTP" in str(w[0].message)

    def test_https_with_token_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_http_with_token("https://api.example.com", has_token=True)
            assert len(w) == 0

    def test_http_without_token_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_http_with_token("http://api.example.com", has_token=False)
            assert len(w) == 0

    def test_https_without_token_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_http_with_token("https://api.example.com", has_token=False)
            assert len(w) == 0


# ==================== validate_enum_param ====================


class TestValidateEnumParam:
    """Tests for validate_enum_param."""

    def test_none_is_allowed(self):
        validate_enum_param(None, VALID_HORIZONS, "horizon")  # should not raise

    def test_valid_value(self):
        validate_enum_param("day", VALID_HORIZONS, "horizon")  # should not raise

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid horizon 'weekly'"):
            validate_enum_param("weekly", VALID_HORIZONS, "horizon")

    def test_error_lists_valid_values(self):
        with pytest.raises(ValueError, match="must be one of"):
            validate_enum_param("X", VALID_METEO_TYPES, "meteo_type")

    def test_case_sensitive(self):
        with pytest.raises(ValueError, match="Invalid snow_type 'hs'"):
            validate_enum_param("hs", VALID_SNOW_TYPES, "snow_type")


# ==================== truncate_response_text ====================


class TestTruncateResponseText:
    """Tests for truncate_response_text."""

    def test_short_text_unchanged(self):
        text = "Short response"
        assert truncate_response_text(text) == text

    def test_exactly_max_length(self):
        text = "x" * 500
        assert truncate_response_text(text) == text

    def test_long_text_truncated(self):
        text = "x" * 600
        result = truncate_response_text(text)
        assert len(result) == 500 + len("... [truncated]")
        assert result.endswith("... [truncated]")

    def test_custom_max_length(self):
        text = "hello world"
        result = truncate_response_text(text, max_length=5)
        assert result == "hello... [truncated]"

    def test_empty_string(self):
        assert truncate_response_text("") == ""


# ==================== safe_int_conversion ====================


class TestSafeIntConversion:
    """Tests for safe_int_conversion."""

    def test_integer_passthrough(self):
        assert safe_int_conversion(42, "field") == 42

    def test_float_to_int(self):
        assert safe_int_conversion(3.0, "field") == 3

    def test_none_returns_none(self):
        assert safe_int_conversion(None, "field") is None

    def test_nan_returns_none(self):
        assert safe_int_conversion(float("nan"), "field") is None

    def test_pandas_nat_returns_none(self):
        assert safe_int_conversion(pd.NaT, "field") is None

    def test_string_number_converts(self):
        assert safe_int_conversion("7", "field") == 7

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot convert field value 'abc'"):
            safe_int_conversion("abc", "field")

    def test_error_includes_field_name(self):
        with pytest.raises(ValueError, match="horizon_value"):
            safe_int_conversion("bad", "horizon_value")
