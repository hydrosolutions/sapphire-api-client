"""
Input validation functions and constants for the SAPPHIRE API client.
"""

import warnings
from typing import Any, Optional, Set
from urllib.parse import urlparse

import pandas as pd

# Valid enum values for API parameters
VALID_HORIZONS: Set[str] = {"day", "pentad", "decade", "month", "season", "year"}
VALID_METEO_TYPES: Set[str] = {"T", "P"}
VALID_SNOW_TYPES: Set[str] = {"HS", "ROF", "SWE"}
VALID_FORECAST_MODELS: Set[str] = {"TFT", "TiDE", "TSMixer", "LR", "EM", "NE"}


def validate_base_url(url: str) -> None:
    """Validate that a URL has a valid scheme and non-empty netloc.

    Args:
        url: The URL to validate.

    Raises:
        ValueError: If the URL scheme is not http/https or netloc is empty.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Invalid URL scheme '{parsed.scheme}': must be 'http' or 'https'"
        )
    if not parsed.netloc:
        raise ValueError(f"Invalid URL '{url}': missing host")


def validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer (> 0).

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative_int(value: int, name: str) -> None:
    """Validate that a value is a non-negative integer (>= 0).

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def warn_http_with_token(url: str, has_token: bool) -> None:
    """Issue a warning if an auth token is being sent over plain HTTP.

    Args:
        url: The base URL.
        has_token: Whether an auth token is configured.
    """
    if has_token and urlparse(url).scheme == "http":
        warnings.warn(
            "Sending auth token over plain HTTP. "
            "Consider using HTTPS in production.",
            UserWarning,
            stacklevel=3,
        )


def validate_enum_param(
    value: Optional[str], valid_values: Set[str], name: str
) -> None:
    """Validate that an optional string parameter is in a set of allowed values.

    Args:
        value: The value to validate (None is allowed and skipped).
        valid_values: Set of valid values.
        name: Parameter name for error messages.

    Raises:
        ValueError: If value is not None and not in valid_values.
    """
    if value is not None and value not in valid_values:
        sorted_values = sorted(valid_values)
        raise ValueError(
            f"Invalid {name} '{value}': must be one of {sorted_values}"
        )


def truncate_response_text(text: str, max_length: int = 500) -> str:
    """Truncate response text to prevent large payloads in exceptions.

    Args:
        text: The response text to truncate.
        max_length: Maximum length (default 500).

    Returns:
        Truncated text with '... [truncated]' suffix if it exceeds max_length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def safe_int_conversion(value: Any, field_name: str) -> Optional[int]:
    """Safely convert a value to int, returning None for NaN/None.

    Args:
        value: The value to convert.
        field_name: Field name for error messages.

    Returns:
        The integer value, or None if the input is NaN/None.

    Raises:
        ValueError: If the value cannot be converted to int.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot convert {field_name} value {value!r} to integer"
        )
