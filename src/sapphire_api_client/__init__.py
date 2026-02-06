"""
SAPPHIRE API Client

Python client for the SAPPHIRE Forecast Tools API.
"""

from sapphire_api_client.client import SapphireAPIClient, SapphireAPIError
from sapphire_api_client.preprocessing import SapphirePreprocessingClient
from sapphire_api_client.postprocessing import SapphirePostprocessingClient
from sapphire_api_client.validators import (
    VALID_FORECAST_MODELS,
    VALID_HORIZONS,
    VALID_METEO_TYPES,
    VALID_SNOW_TYPES,
)

__version__ = "0.1.0"

__all__ = [
    "SapphireAPIClient",
    "SapphireAPIError",
    "SapphirePreprocessingClient",
    "SapphirePostprocessingClient",
    "VALID_FORECAST_MODELS",
    "VALID_HORIZONS",
    "VALID_METEO_TYPES",
    "VALID_SNOW_TYPES",
]
