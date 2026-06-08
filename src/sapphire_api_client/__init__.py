"""
SAPPHIRE API Client

Python client for the SAPPHIRE Forecast Tools API.
"""

from sapphire_api_client.client import SapphireAPIClient, SapphireAPIError
from sapphire_api_client.long_term import SapphireLongTermForecastClient
from sapphire_api_client.postprocessing import SapphirePostprocessingClient
from sapphire_api_client.preprocessing import SapphirePreprocessingClient
from sapphire_api_client.short_term import SapphireShortTermForecastClient
from sapphire_api_client.validators import (
    VALID_FORECAST_MODELS,
    VALID_HORIZONS,
    VALID_METEO_TYPES,
    VALID_SKILL_METRIC_HORIZONS,
    VALID_SNOW_TYPES,
)

__version__ = "0.4.0"

__all__ = [
    "SapphireAPIClient",
    "SapphireAPIError",
    "SapphireLongTermForecastClient",
    "SapphirePostprocessingClient",
    "SapphirePreprocessingClient",
    "SapphireShortTermForecastClient",
    "VALID_FORECAST_MODELS",
    "VALID_HORIZONS",
    "VALID_METEO_TYPES",
    "VALID_SKILL_METRIC_HORIZONS",
    "VALID_SNOW_TYPES",
]
