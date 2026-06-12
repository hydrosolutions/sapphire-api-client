"""
Base class for SAPPHIRE Postprocessing API clients.

Holds the SERVICE_PREFIX and skill metric methods shared by all
postprocessing forecast families.
"""

import logging
import warnings
from datetime import date
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from sapphire_api_client.client import SapphireAPIClient
from sapphire_api_client.validators import (
    HorizonTypeLiteral,
    VALID_SKILL_METRIC_HORIZONS,
    validate_enum_param,
    validate_non_negative_int,
    validate_positive_int,
)

logger = logging.getLogger(__name__)


class SapphirePostprocessingBase(SapphireAPIClient):
    """
    Base client for the SAPPHIRE Postprocessing API.

    Provides the service prefix and skill metric methods shared by all
    postprocessing forecast clients.
    """

    # Service prefix for API gateway routing
    SERVICE_PREFIX = "/api/postprocessing"

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
        validate_enum_param(horizon, VALID_SKILL_METRIC_HORIZONS, "horizon")
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
        horizon_type: HorizonTypeLiteral,
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
