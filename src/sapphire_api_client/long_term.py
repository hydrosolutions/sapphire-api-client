"""
Client for SAPPHIRE long-term forecasts.
"""

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from sapphire_api_client.postprocessing_base import SapphirePostprocessingBase
from sapphire_api_client.validators import (
    VALID_LONG_FORECAST_HORIZONS,
    VALID_LONG_FORECAST_MODELS,
    validate_enum_param,
    validate_non_negative_int,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class SapphireLongTermForecastClient(SapphirePostprocessingBase):
    """
    Client for long-term forecasts (monthly, quarterly, seasonal).

    Long-term forecasts include quantile predictions (Q05-Q95) and a
    validity period (valid_from/valid_to).

    Example:
        >>> client = SapphireLongTermForecastClient(
        ...     base_url="http://localhost:8000", auth_token="..."
        ... )
        >>> df = client.read_long_term_forecasts(code="15013")
    """

    # ==================== LONG-TERM FORECASTS ====================

    def read_long_term_forecasts(
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
        Read long-term forecast data from the API.

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

    def write_long_term_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """
        Write long-term forecast records to the API.

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
    def prepare_long_term_forecast_records(
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
        Prepare long-term forecast records from a DataFrame.

        Converts a DataFrame to a list of dicts ready for
        write_long_term_forecasts(). Handles NaN-to-None conversion for all
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
            List of record dicts ready for write_long_term_forecasts()

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
