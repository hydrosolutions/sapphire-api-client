# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install for development
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_client.py

# Run a specific test
uv run pytest tests/test_client.py::TestSapphireAPIClient::test_health_check_success

# Run with coverage
uv run pytest --cov=src/sapphire_api_client

# Type checking
uv run mypy
```

## Type Checking

This package uses strict type checking with mypy. All code must pass `uv run mypy` before merging.

Key rules:
- All functions must have type annotations
- Use `Dict[str, Any]` for dicts with mixed value types (e.g., API params, records)
- Use `cast()` when the return type is known but mypy can't infer it (e.g., `response.json()`)
- The package exports type information via `py.typed` (PEP 561)

## Architecture

This is a Python client library for the SAPPHIRE Forecast Tools API, used for hydrological data management.

### Client Hierarchy

```
SapphireAPIClient (base)           # src/sapphire_api_client/client.py - retry logic, batching, HTTP methods
├── SapphirePreprocessingClient    # src/sapphire_api_client/preprocessing.py - runoff, hydrograph, meteo, snow
└── SapphirePostprocessingClient   # src/sapphire_api_client/postprocessing.py - forecasts, LR forecasts, skill metrics
```

The base client handles:
- Automatic retry with exponential backoff (tenacity) for transient failures (502, 503, 504)
- Batched POST requests via `_post_batched()` for large datasets
- Common `_get()`, `_post()`, `_make_request()` methods

### Data Flow Pattern

Each data type follows this pattern:
1. `read_*()` - GET from API, returns pandas DataFrame
2. `write_*()` - POST records to API (uses batching)
3. `prepare_*_records()` - Static method to convert DataFrame to API-ready dicts

### Package Structure

- Source code is in `src/sapphire_api_client/` (src layout)
- Tests use `responses` library to mock HTTP requests
- Exports: `SapphireAPIClient`, `SapphireAPIError`, `SapphirePreprocessingClient`, `SapphirePostprocessingClient`

### Data Types

**Preprocessing** (input data):
- Runoff: daily discharge time series with horizons (day, pentad, decade, month, season, year)
- Hydrograph: statistical summaries (mean, std, quantiles, norm)
- Meteo: temperature (T) and precipitation (P)
- Snow: height (HS), SWE, runoff (ROF) with zone values (value1-value14)

**Postprocessing** (output data):
- Forecasts: predictions with confidence bounds
- LR Forecasts: linear regression forecasts
- Skill Metrics: mae, rmse, nse, kge, bias, r2, pbias
