# SAPPHIRE API Client

[![Tests](https://github.com/hydrosolutions/sapphire-api-client/actions/workflows/test.yml/badge.svg)](https://github.com/hydrosolutions/sapphire-api-client/actions/workflows/test.yml)

## Maintenance Status

🟢 **Active** – Ongoing development as part of [SAPPHIRE Forecast Tools](https://github.com/hydrosolutions/sapphire-forecast-tools)

---

Python client for the SAPPHIRE Forecast Tools API. This package provides a convenient interface for reading from and writing to the SAPPHIRE preprocessing and postprocessing databases.

## Installation

```bash
uv pip install git+https://github.com/hydrosolutions/sapphire-api-client.git
```

## Usage

### Preprocessing Client

```python
from sapphire_api_client import SapphirePreprocessingClient
import pandas as pd

# Initialize client (defaults to localhost:8000)
client = SapphirePreprocessingClient()

# Or specify a different base URL
client = SapphirePreprocessingClient(base_url="https://api.example.com")

# Read runoff data
df = client.read_runoff(
    horizon="day",
    code="12345",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# Write runoff data from DataFrame
records = SapphirePreprocessingClient.prepare_runoff_records(
    df=my_dataframe,
    horizon_type="day",
    code="12345"
)
count = client.write_runoff(records)
print(f"Wrote {count} records")
```

### Postprocessing Client

```python
from sapphire_api_client import SapphirePostprocessingClient

client = SapphirePostprocessingClient()

# Read forecasts
forecasts = client.read_forecasts(
    horizon="pentad",
    code="12345"
)

# Write forecasts
records = SapphirePostprocessingClient.prepare_forecast_records(df, ...)
client.write_forecasts(records)
```

## Features

- Automatic retry with exponential backoff on transient failures
- Batch posting for large datasets
- Strict error handling (fails fast on persistent errors)
- Type-safe record preparation from pandas DataFrames
- Support for all SAPPHIRE data types:
  - Runoff (daily time series)
  - Hydrograph (statistical summaries)
  - Meteorological data (temperature, precipitation)
  - Snow data (height, SWE, runoff)
  - Forecasts and skill metrics

## Configuration

The client can be configured with:

- `base_url`: API base URL (default: `http://localhost:8000`)
- `auth_token`: Optional Bearer token for authentication
- `max_retries`: Maximum retry attempts (default: 3)
- `batch_size`: Records per batch for bulk writes (default: 1000)

## Authentication

The client supports Bearer token authentication for controlled access.

**Security best practice:** Store tokens in environment variables, never hardcode them.

```python
import os

# Load token from environment variable
auth_token = os.environ.get("SAPPHIRE_API_TOKEN")

# Hydromet with full read/write access
client = SapphirePreprocessingClient(
    base_url="https://api.hydromet.example.org",
    auth_token=auth_token,
)

# Check if client is authenticated
if client.is_authenticated:
    print("Using authenticated access")
```

**Access levels (configured server-side):**
- **Hydromet operators**: Full read/write access to all data
- **External organizations**: Read-only access to forecasts for specific sites/horizons

The server-side auth service controls which resources each token can access.

## Error Handling

The client raises `SapphireAPIError` on failures:

```python
from sapphire_api_client import SapphireAPIError

try:
    client.write_runoff(records)
except SapphireAPIError as e:
    print(f"API error: {e}")
    # Handle failure - the operation should be retried or investigated
```

## Contributing

```bash
git clone https://github.com/hydrosolutions/sapphire-api-client.git
cd sapphire-api-client
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run tests:

```bash
uv run pytest
```

## License

MIT License. See the [LICENSE](LICENSE) file for details.
