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
SapphireAPIClient (base)                    # src/sapphire_api_client/client.py - retry logic, batching, HTTP methods
├── SapphirePreprocessingClient             # src/sapphire_api_client/preprocessing.py - runoff, hydrograph, meteo, snow
└── SapphirePostprocessingBase              # src/sapphire_api_client/postprocessing_base.py - SERVICE_PREFIX + skill metrics
    ├── SapphireShortTermForecastClient     # src/sapphire_api_client/short_term.py - short-term forecasts + LR forecasts
    ├── SapphireLongTermForecastClient      # src/sapphire_api_client/long_term.py - long-term forecasts
    └── SapphirePostprocessingClient        # src/sapphire_api_client/postprocessing.py - facade (inherits ShortTerm + LongTerm, deprecated aliases)
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
- Exports: `SapphireAPIClient`, `SapphireAPIError`, `SapphirePreprocessingClient`, `SapphirePostprocessingClient`, `SapphireShortTermForecastClient`, `SapphireLongTermForecastClient`

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

## Testing Requirements

### Testing Philosophy

Good tests describe contracts — what must stay true even if the implementation changes. If a test breaks after a refactor that doesn't change behavior, the test was wrong.

### Golden Rules

1. **Test behavior, not implementation** — assert on outputs and public APIs. Do not inspect private attributes (`._internal_state`) unless no public API exists.
2. **Prefer fast, deterministic tests** — no `sleep()`, no uncontrolled `datetime.now()` or `random`. Pass dates and other non-deterministic values as parameters (see "Deterministic Time" below).
3. **Use fakes over mocks where practical** — a fake implementation (e.g., an in-memory store, a temp directory with real files) is easier to read and more resilient than a chain of `MagicMock` assertions. Reserve `MagicMock` for external boundaries (API clients, file I/O, external services).
4. **Structure tests as Arrange → Act → Assert** — setup the data, call the function, check the result. Name fixtures descriptively (`config_with_missing_fields`, `df_with_nan_values`), not generically (`data`, `fixture1`).

### Deterministic Time

Never use `datetime.now()` or `date.today()` directly in business logic. Capture the current date/time once at the entry point and pass it as a parameter to all downstream functions.

**Why this matters:**

1. **Clock-tick bug:** If a pipeline runs across midnight, functions that independently call `date.today()` disagree on what "today" is.
2. **Default argument bug:** `def f(year=datetime.now().year)` is evaluated once at import time — if the module is imported before midnight on Dec 31 and the function is called after, `year` is silently wrong.
3. **Replay/backtest support:** When re-running logic for historical dates, scattered `date.today()` calls return the current date instead of the target date.

**Pattern — capture once, pass everywhere:**

```python
# Entry point (main.py, CLI, orchestrator):
run_date = date.today()  # single source of truth

# All downstream functions receive it as a parameter:
def process_data(data: pd.DataFrame, run_date: date) -> pd.DataFrame:
    current_year = run_date.year
    ...

# NEVER in a default argument:
# WRONG — evaluated once at import time
def get_period(period_id, year=datetime.now().year): ...

# CORRECT — caller passes explicitly
def get_period(period_id: int, year: int) -> date: ...
```

**Acceptable uses of `datetime.now()`:** Logging timestamps, file naming, and performance timers — these should reflect actual wall-clock time.

**Testing benefit:** Boundary-date testing becomes trivial — no datetime mocking or freezegun needed:

```python
def test_year_boundary():
    result = process(run_date=date(2025, 12, 31), ...)
    assert result.year == 2025

def test_leap_year():
    result = process(run_date=date(2024, 2, 29), ...)
    assert result.day_of_year == 60
```

### Before Committing

All tests must pass with zero skips before committing or moving to a new topic.

**Zero Skips Policy**

No tests may be skipped without justification. If any tests are skipped or fail to collect, treat this as a red flag requiring investigation before proceeding. Do not accept "0 collected" or `pytest.skip()` as normal — find and fix the root cause.

Dependency-gated skips are acceptable when an optional dependency is not installed. These must guard on a module-level flag and skip with an explicit message (e.g., `pytest.skip("optional-lib not installed")`).

### Test Categories

Every new feature or bug fix must include tests. The required categories depend on what changed:

#### 1. Unit Tests (always required)

Isolated tests for individual functions with all external dependencies mocked. Each new or modified public function needs at least:
- A happy-path test with typical input
- An error-path test (invalid input, exception handling)

For error-path tests, always assert both exception type and message fragment:

```python
with pytest.raises(ValueError, match="must be positive"):
    calculate(data, horizon=-1)
```

#### Assertion Quality

Tests must verify correctness, not just existence. Weak assertions let bugs pass silently.

Rules:
- Use exact counts (`assert len(rows) == 2`) not vague checks (`assert len(rows) > 0`)
- Spot-check at least one record's values (e.g., `assert record['value'] == 105.0`)
- For DataFrames, prefer `pd.testing.assert_frame_equal` over row-count comparisons
- For API records or dicts, verify field values (not just key existence) for at least one representative record
- Avoid ambiguous `or` in assertions (`assert x.empty or 'FOO' not in x` can mask bugs — be explicit about which condition you expect)

#### 2. Edge Case Tests (required for DataFrame, date, or numeric code)

Any code that processes DataFrames, dates, or numeric values must have edge case tests covering these scenarios:

| Category | Scenarios to test |
|---|---|
| Empty data | Empty DataFrame, single-row DataFrame, all-NaN columns |
| NaN handling | All NaN values, mixed NaN/valid, NaN-to-None conversion for API |
| Date boundaries | Year transitions (Dec 31 → Jan 1), leap year Feb 29, month boundaries |
| Value boundaries | Zero values, very small positives (0.001), very large values (10000+) |
| Duplicates | Duplicate key combinations |
| Multi-entity | Single entity many dates, many entities single date |
| Data preservation | Non-transformed columns, schema, and row order remain intact after processing |

#### 3. Integration Tests (required for multi-step workflows)

Tests that exercise the real logic across multiple internal functions, only mocking external boundaries (API clients, file I/O). Required when:
- A function calls multiple internal modules in sequence
- Data flows through a pipeline (read → transform → write)
- Entry points orchestrate multiple steps

Integration tests should:
- Use real logic for everything inside the boundary
- Only mock external services and filesystem — prefer fakes (e.g., a temp directory with real files) over `MagicMock` chains for file I/O
- Validate the full data flow, not just final output — check intermediate state at each pipeline stage
- Verify data preservation: columns not touched by the pipeline must survive unchanged

File naming: `test_integration_<topic>.py`

#### 4. External Service Failure Tests (required for any code calling external APIs)

Any function that reads from or writes to an external service must have tests for all failure modes:

```python
class TestWriteToApi:
    def test_returns_false_when_client_unavailable(self, data):
        """When the client library is not installed."""
        ...

    def test_returns_false_when_service_disabled(self, data):
        """When the service is disabled via config/env."""
        ...

    def test_returns_false_when_service_not_ready(self, data):
        """When the readiness/health check fails."""
        ...

    def test_fallback_still_works_on_api_failure(self, data, tmp_path):
        """Fallback path works when the service raises."""
        ...
```

#### 5. Performance Benchmarks (optional, for optimization work)

Mark with `@pytest.mark.benchmark`. Skipped by default, run explicitly:

```bash
pytest tests/test_performance.py -v -k bench
```

### Test File Naming Conventions

| File name pattern | Contents |
|---|---|
| `test_<topic>.py` | Unit tests for a specific topic or module |
| `test_edge_cases.py` | Edge case and boundary condition tests |
| `test_integration_<topic>.py` | Multi-step workflow integration tests |
| `test_performance.py` | Performance benchmarks (`@pytest.mark.benchmark`) |

### Test Anti-Patterns (avoid these)

- Asserting on private attributes (`._steps`, `._internal_cache`) — test public behavior instead
- Giant integration tests covering all cases — push variation into unit tests; integration tests cover the happy-path pipeline and one or two failure modes
- Hiding critical setup in deeply nested fixtures — if a test is hard to understand without reading three conftest files, flatten the setup
- Bare `except:` in test helpers — let unexpected exceptions propagate so they surface as test failures
- `MagicMock` chains for internal modules — if you're mocking three internal functions to test a fourth, the test is too coupled to implementation; restructure or test at a higher level
- Tests that pass regardless of correctness — e.g., `assert len(result) > 0` when the function could return garbage rows. Always verify values, not just shapes (see Assertion Quality above)
- Non-deterministic time dependence — tests that break on Jan 1 or Feb 29 because they call `date.today()` instead of receiving the date as a parameter
- `datetime.now()` in default arguments — `def f(year=datetime.now().year)` is evaluated once at import time and goes stale at year boundaries; always require explicit arguments
