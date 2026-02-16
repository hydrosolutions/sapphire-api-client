"""
Client for SAPPHIRE short-term forecasts and linear regression forecasts.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from sapphire_api_client.postprocessing_base import SapphirePostprocessingBase
from sapphire_api_client.validators import (
    VALID_FORECAST_MODELS,
    VALID_HORIZONS,
    validate_enum_param,
    validate_non_negative_int,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class SapphireShortTermForecastClient(SapphirePostprocessingBase):
    """
    Client for short-term forecasts and linear regression forecasts.

    Provides methods for reading and writing:
    - Short-term forecasts (with confidence bounds)
    - Linear regression (LR) forecasts

    Example:
        >>> client = SapphireShortTermForecastClient(
        ...     base_url="http://localhost:8000", auth_token="..."
        ... )
        >>> df = client.read_short_term_forecasts(horizon="pentad", code="12345")
    """

    # ==================== SHORT-TERM FORECASTS ====================

    def read_short_term_forecasts(
        self,
        horizon: Optional[str] = None,
        code: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        target: Optional[Union[str, date]] = None,
        start_target: Optional[Union[str, date]] = None,
        end_target: Optional[Union[str, date]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read short-term forecast data from the API.

        Args:
            horizon: Horizon type filter
            code: Station code filter
            model: Model type filter (TFT, TiDE, TSMixer, LR, EM, NE)
            start_date: Start date filter (forecast issue date)
            end_date: End date filter (forecast issue date)
            target: Target date filter (the date the forecast is for)
            start_target: Start of target date range filter
            end_target: End of target date range filter
            skip: Pagination offset
            limit: Maximum records

        Returns:
            DataFrame with forecast data
        """
        validate_enum_param(horizon, VALID_HORIZONS, "horizon")
        validate_enum_param(model, VALID_FORECAST_MODELS, "model")
        validate_non_negative_int(skip, "skip")
        validate_positive_int(limit, "limit")

        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if horizon:
            params["horizon"] = horizon
        if code:
            params["code"] = code
        if model:
            params["model"] = model
        if start_date:
            params["start_date"] = str(start_date)
        if end_date:
            params["end_date"] = str(end_date)
        if target:
            params["target"] = str(target)
        if start_target:
            params["start_target"] = str(start_target)
        if end_target:
            params["end_target"] = str(end_target)

        logger.info("Reading forecasts (horizon=%s, code=%s, model=%s)", horizon, code, model)
        records = self._get("/forecast/", params=params)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_short_term_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """
        Write short-term forecast records to the API.

        Args:
            records: List of forecast records

        Returns:
            Number of records written
        """
        return self._post_batched("/forecast/", records)

    @staticmethod
    def prepare_short_term_forecast_records(
        df: pd.DataFrame,
        horizon_type: Literal["day", "pentad", "decade", "month", "season", "year"],
        code: str,
        date_col: str = "date",
        forecast_col: str = "forecast",
        lower_col: Optional[str] = "lower",
        upper_col: Optional[str] = "upper",
    ) -> List[Dict[str, Any]]:
        """
        Prepare short-term forecast records from a DataFrame.

        Args:
            df: Source DataFrame
            horizon_type: Horizon type
            code: Station code
            date_col: Name of date column
            forecast_col: Name of forecast value column
            lower_col: Name of lower bound column (optional)
            upper_col: Name of upper bound column (optional)

        Returns:
            List of records ready for API
        """
        required_cols = [date_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        records = []
        for _, row in df.iterrows():
            record: Dict[str, Any] = {
                "horizon_type": horizon_type,
                "code": code,
                "date": str(row[date_col]),
            }
            # Forecast value
            val = row.get(forecast_col)
            record["forecast"] = val if pd.notna(val) else None

            # Confidence bounds
            if lower_col and lower_col in df.columns:
                lower_val = row.get(lower_col)
                record["lower"] = lower_val if pd.notna(lower_val) else None

            if upper_col and upper_col in df.columns:
                upper_val = row.get(upper_col)
                record["upper"] = upper_val if pd.notna(upper_val) else None

            records.append(record)
        return records

    # ==================== LR FORECASTS ====================

    def read_lr_forecasts(
        self,
        horizon: Optional[str] = None,
        code: Optional[str] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read linear regression forecast data from the API.

        Args:
            horizon: Horizon type filter
            code: Station code filter
            start_date: Start date filter
            end_date: End date filter
            skip: Pagination offset
            limit: Maximum records

        Returns:
            DataFrame with LR forecast data
        """
        validate_enum_param(horizon, VALID_HORIZONS, "horizon")
        validate_non_negative_int(skip, "skip")
        validate_positive_int(limit, "limit")

        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if horizon:
            params["horizon"] = horizon
        if code:
            params["code"] = code
        if start_date:
            params["start_date"] = str(start_date)
        if end_date:
            params["end_date"] = str(end_date)

        logger.info("Reading LR forecasts (horizon=%s, code=%s)", horizon, code)
        records = self._get("/lr-forecast/", params=params)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_lr_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """
        Write linear regression forecast records to the API.

        Args:
            records: List of LR forecast records

        Returns:
            Number of records written
        """
        return self._post_batched("/lr-forecast/", records)
