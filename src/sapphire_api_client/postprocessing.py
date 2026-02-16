"""
Client for SAPPHIRE Postprocessing API.

Handles forecasts, linear regression forecasts, and skill metrics.
"""

import logging
import warnings
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from sapphire_api_client.client import SapphireAPIClient
from sapphire_api_client.validators import (
    VALID_HORIZONS,
    VALID_LONG_FORECAST_HORIZONS,
    VALID_LONG_FORECAST_MODELS,
    validate_enum_param,
    validate_non_negative_int,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class SapphirePostprocessingClient(SapphireAPIClient):
    """
    Client for the SAPPHIRE Postprocessing API.

    Provides methods for reading and writing:
    - Forecasts
    - Linear regression forecasts
    - Skill metrics
    - Long forecasts (monthly/quarterly/seasonal with quantiles)

    Note: The postprocessing API endpoints may differ from preprocessing.
    Update this client as the postprocessing service is developed.

    Example:
        >>> client = SapphirePostprocessingClient()
        >>> df = client.read_forecasts(horizon="pentad", code="12345")
    """

    # Service prefix for API gateway routing
    SERVICE_PREFIX = "/api/postprocessing"

    # ==================== FORECASTS ====================

    def read_forecasts(
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
        Read forecast data from the API.

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

    def write_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """
        Write forecast records to the API.

        Args:
            records: List of forecast records

        Returns:
            Number of records written
        """
        return self._post_batched("/forecast/", records)

    @staticmethod
    def prepare_forecast_records(
        df: pd.DataFrame,
        horizon_type: Literal["day", "pentad", "decade", "month", "season", "year"],
        code: str,
        date_col: str = "date",
        forecast_col: str = "forecast",
        lower_col: Optional[str] = "lower",
        upper_col: Optional[str] = "upper",
    ) -> List[Dict[str, Any]]:
        """
        Prepare forecast records from a DataFrame.

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

    # ==================== SKILL METRICS ====================

    def read_skill_metrics(
        self,
        horizon: Optional[str] = None,
        code: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read skill metrics from the API.

        Args:
            horizon: Horizon type filter
            code: Station code filter
            model: Model name filter
            start_date: Start date filter for skill metrics
            end_date: End date filter for skill metrics
            skip: Pagination offset
            limit: Maximum records

        Returns:
            DataFrame with skill metrics
        """
        validate_enum_param(horizon, VALID_HORIZONS, "horizon")
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

        logger.info("Reading skill metrics (horizon=%s, code=%s, model=%s)", horizon, code, model)
        records = self._get("/skill-metric/", params=params)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_skill_metrics(self, records: List[Dict[str, Any]]) -> int:
        """
        Write skill metric records to the API.

        Args:
            records: List of skill metric records

        Returns:
            Number of records written
        """
        return self._post_batched("/skill-metric/", records)

    @staticmethod
    def prepare_skill_metric_records(
        df: pd.DataFrame,
        horizon_type: Literal["day", "pentad", "decade", "month", "season", "year"],
        code: str,
        model: str,
    ) -> List[Dict[str, Any]]:
        """
        Prepare skill metric records from a DataFrame.

        Expects columns for metrics like: mae, rmse, nse, kge, bias, etc.

        Args:
            df: Source DataFrame
            horizon_type: Horizon type
            code: Station code
            model: Model name

        Returns:
            List of records ready for API
        """
        metric_cols = ["mae", "rmse", "nse", "kge", "bias", "r2", "pbias"]

        found_metrics = [c for c in metric_cols if c in df.columns]
        if not found_metrics:
            warnings.warn(
                f"No metric columns found in DataFrame. "
                f"Expected at least one of: {metric_cols}",
                UserWarning,
                stacklevel=2,
            )

        records = []
        for _, row in df.iterrows():
            record: Dict[str, Any] = {
                "horizon_type": horizon_type,
                "code": code,
                "model": model,
            }
            # Add metric columns
            for col in metric_cols:
                if col in df.columns:
                    val = row.get(col)
                    record[col] = val if pd.notna(val) else None
            records.append(record)
        return records

    # ==================== LONG FORECASTS ====================

    def read_long_forecasts(
        self,
        horizon_type: Optional[str] = None,
        horizon_value: Optional[int] = None,
        code: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        valid_from: Optional[Union[str, date]] = None,
        valid_to: Optional[Union[str, date]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Read long forecast data from the API.

        Long forecasts are produced by the long_term_forecasting module and
        stored in the long_forecasts table. They include quantile predictions
        (Q05-Q95) and a validity period (valid_from/valid_to).

        Args:
            horizon_type: Horizon type filter
                ("month", "quarter", "season")
            horizon_value: Horizon value filter (e.g., 1-12 for months)
            code: Station code filter
            model: Model type filter (GBT, LR_Base, SM_GBT, MC_ALD, etc.)
            start_date: Start date filter for forecast issue date (inclusive)
            end_date: End date filter for forecast issue date (inclusive)
            valid_from: Filter: valid_from >= this value
            valid_to: Filter: valid_to <= this value
            skip: Pagination offset
            limit: Maximum records

        Returns:
            DataFrame with long forecast data. Empty DataFrame if no records.
            Columns include: horizon_type, horizon_value, code, date,
            model_type, valid_from, valid_to, flag, composition,
            q, q_obs, q_xgb, q_lgbm, q_catboost, q_loc,
            q05, q10, q25, q50, q75, q90, q95,
            id, model_type_description
        """
        validate_enum_param(horizon_type, VALID_LONG_FORECAST_HORIZONS, "horizon_type")
        validate_enum_param(model, VALID_LONG_FORECAST_MODELS, "model")
        validate_non_negative_int(skip, "skip")
        validate_positive_int(limit, "limit")

        params: Dict[str, Any] = {"skip": skip, "limit": limit}
        if horizon_type:
            params["horizon_type"] = horizon_type
        if horizon_value is not None:
            params["horizon_value"] = horizon_value
        if code:
            params["code"] = code
        if model:
            params["model"] = model
        if start_date:
            params["start_date"] = str(start_date)
        if end_date:
            params["end_date"] = str(end_date)
        if valid_from:
            params["valid_from"] = str(valid_from)
        if valid_to:
            params["valid_to"] = str(valid_to)

        logger.info(
            "Reading long forecasts (horizon_type=%s, code=%s, model=%s)",
            horizon_type, code, model,
        )
        records = self._get("/long-forecast/", params=params)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def write_long_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """
        Write long forecast records to the API.

        Records are upserted based on the unique key:
        (horizon_type, horizon_value, code, date, model_type, valid_from,
        valid_to).

        Args:
            records: List of long forecast record dicts. Required keys:
                horizon_type, horizon_value, code, date, model_type,
                valid_from, valid_to. Optional keys: flag, composition,
                q, q_obs, q_xgb, q_lgbm, q_catboost, q_loc,
                q05, q10, q25, q50, q75, q90, q95.

        Returns:
            Number of records written
        """
        return self._post_batched("/long-forecast/", records)

    @staticmethod
    def prepare_long_forecast_records(
        df: pd.DataFrame,
        horizon_type: str,
        horizon_value: int,
        model_type: str,
        code_col: str = "code",
        date_col: str = "date",
        valid_from_col: str = "valid_from",
        valid_to_col: str = "valid_to",
    ) -> List[Dict[str, Any]]:
        """
        Prepare long forecast records from a DataFrame.

        Converts a DataFrame to a list of dicts ready for
        write_long_forecasts(). Handles NaN-to-None conversion for all
        nullable float fields.

        Args:
            df: Source DataFrame with forecast data
            horizon_type: Horizon type ("month", "quarter", "season")
            horizon_value: Which period (e.g., month number 1-12)
            model_type: Model type string (e.g., "GBT", "LR_Base")
            code_col: Column name for station code
            date_col: Column name for forecast issue date
            valid_from_col: Column name for validity start date
            valid_to_col: Column name for validity end date

        Returns:
            List of record dicts ready for write_long_forecasts()

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        required_cols = [code_col, date_col, valid_from_col, valid_to_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        quantile_cols = [
            "q", "q_obs", "q_xgb", "q_lgbm", "q_catboost", "q_loc",
            "q05", "q10", "q25", "q50", "q75", "q90", "q95",
        ]

        records = []
        for _, row in df.iterrows():
            record: Dict[str, Any] = {
                "horizon_type": horizon_type,
                "horizon_value": horizon_value,
                "code": str(row[code_col]),
                "date": str(row[date_col]),
                "model_type": model_type,
                "valid_from": str(row[valid_from_col]),
                "valid_to": str(row[valid_to_col]),
            }

            # Optional fields
            if "flag" in df.columns:
                val = row.get("flag")
                record["flag"] = int(val) if pd.notna(val) else None
            if "composition" in df.columns:
                val = row.get("composition")
                record["composition"] = val if pd.notna(val) else None

            # Quantile predictions — NaN→None
            for col in quantile_cols:
                if col in df.columns:
                    val = row.get(col)
                    record[col] = float(val) if pd.notna(val) else None

            records.append(record)
        return records
