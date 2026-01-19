"""
SAPPHIRE API Client

Python client for the SAPPHIRE Forecast Tools API.
"""

from sapphire_api_client.client import SapphireAPIClient, SapphireAPIError
from sapphire_api_client.preprocessing import SapphirePreprocessingClient
from sapphire_api_client.postprocessing import SapphirePostprocessingClient

__version__ = "0.1.0"

__all__ = [
    "SapphireAPIClient",
    "SapphireAPIError",
    "SapphirePreprocessingClient",
    "SapphirePostprocessingClient",
]
