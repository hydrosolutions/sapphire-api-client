# Changelog

## [0.3.0] - 2026-02-16

### Added
- Long-term forecast client (`SapphireLongTermForecastClient`) with read, write, and record preparation for monthly, quarterly, and seasonal forecasts with quantile predictions (Q05-Q95)
- Input validation for long-term forecast horizons (`month`, `quarter`, `season`) and 22 forecast model types
- Edge case and validation tests for both short-term and long-term forecast clients

### Changed
- Split postprocessing client into focused domain classes:
  - `SapphireShortTermForecastClient` for short-term and LR forecasts
  - `SapphireLongTermForecastClient` for long-term forecasts
  - `SapphirePostprocessingClient` retained as a facade combining both
- Deprecated old method names on `SapphirePostprocessingClient` (`read_forecasts`, `write_forecasts`, `prepare_forecast_records`, `read_long_forecasts`, `write_long_forecasts`, `prepare_long_forecast_records`) in favor of explicit `*_short_term_*` and `*_long_term_*` variants

## [0.2.0] - 2026-02-06

### Added
- Validators module with input validation for horizons, meteo types, snow types, and forecast models
- URL validation, pagination parameter validation, and enum parameter checks
- Improved error messages with sorted valid values on validation failures

### Changed
- Hardened all client methods with input validation before API calls

## [0.1.0] - 2026-01-19

### Added
- Initial release
- Preprocessing client (runoff, hydrograph, meteo, snow)
- Postprocessing client (forecasts, LR forecasts, skill metrics)
- Bearer token authentication
- Automatic retry with exponential backoff
- Batch posting for large datasets
