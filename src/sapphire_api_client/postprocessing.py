"""
Backwards-compatible facade for SAPPHIRE Postprocessing API.

``SapphirePostprocessingClient`` inherits from both
:class:`SapphireShortTermForecastClient` and
:class:`SapphireLongTermForecastClient`, so it exposes every method from
both forecast families plus the skill-metric methods from the shared base.

Old method names (``read_forecasts``, ``write_forecasts``, etc.) are kept as
deprecated aliases that emit :class:`DeprecationWarning` and delegate to the
renamed methods.
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

from datetime import date

import pandas as pd

from sapphire_api_client.long_term import SapphireLongTermForecastClient
from sapphire_api_client.short_term import SapphireShortTermForecastClient


class SapphirePostprocessingClient(
    SapphireShortTermForecastClient,
    SapphireLongTermForecastClient,
):
    """
    Combined client for the SAPPHIRE Postprocessing API.

    Provides access to all postprocessing endpoints:
    - Short-term forecasts (via :class:`SapphireShortTermForecastClient`)
    - Linear regression forecasts (via :class:`SapphireShortTermForecastClient`)
    - Long-term forecasts (via :class:`SapphireLongTermForecastClient`)
    - Skill metrics (via :class:`SapphirePostprocessingBase`)

    For new code, prefer the focused client classes directly. This class
    exists for backwards compatibility with existing consumers.

    Example:
        >>> client = SapphirePostprocessingClient(
        ...     base_url="http://localhost:8000", auth_token="..."
        ... )
        >>> df = client.read_short_term_forecasts(horizon="pentad", code="12345")
    """

    # ==================== DEPRECATED ALIASES ====================

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
        """Deprecated: use :meth:`read_short_term_forecasts` instead."""
        warnings.warn(
            "read_forecasts() is deprecated, use read_short_term_forecasts() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read_short_term_forecasts(
            horizon=horizon,
            code=code,
            model=model,
            start_date=start_date,
            end_date=end_date,
            target=target,
            start_target=start_target,
            end_target=end_target,
            skip=skip,
            limit=limit,
        )

    def write_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """Deprecated: use :meth:`write_short_term_forecasts` instead."""
        warnings.warn(
            "write_forecasts() is deprecated, use write_short_term_forecasts() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_short_term_forecasts(records)

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
        """Deprecated: use :meth:`prepare_short_term_forecast_records` instead."""
        warnings.warn(
            "prepare_forecast_records() is deprecated, "
            "use prepare_short_term_forecast_records() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return SapphireShortTermForecastClient.prepare_short_term_forecast_records(
            df=df,
            horizon_type=horizon_type,
            code=code,
            date_col=date_col,
            forecast_col=forecast_col,
            lower_col=lower_col,
            upper_col=upper_col,
        )

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
        """Deprecated: use :meth:`read_long_term_forecasts` instead."""
        warnings.warn(
            "read_long_forecasts() is deprecated, use read_long_term_forecasts() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read_long_term_forecasts(
            horizon_type=horizon_type,
            horizon_value=horizon_value,
            code=code,
            model=model,
            start_date=start_date,
            end_date=end_date,
            valid_from=valid_from,
            valid_to=valid_to,
            skip=skip,
            limit=limit,
        )

    def write_long_forecasts(self, records: List[Dict[str, Any]]) -> int:
        """Deprecated: use :meth:`write_long_term_forecasts` instead."""
        warnings.warn(
            "write_long_forecasts() is deprecated, use write_long_term_forecasts() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.write_long_term_forecasts(records)

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
        """Deprecated: use :meth:`prepare_long_term_forecast_records` instead."""
        warnings.warn(
            "prepare_long_forecast_records() is deprecated, "
            "use prepare_long_term_forecast_records() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return SapphireLongTermForecastClient.prepare_long_term_forecast_records(
            df=df,
            horizon_type=horizon_type,
            horizon_value=horizon_value,
            model_type=model_type,
            code_col=code_col,
            date_col=date_col,
            valid_from_col=valid_from_col,
            valid_to_col=valid_to_col,
        )
