# Changelog

## [0.5.0] - 2026-11-09

### Added
- Retry-After header support (RFC 7231) for intelligent rate limit handling
- `parse_retry_after()` function supporting both seconds and HTTP-date formats
- Automatic cooldown adjustment based on server-provided Retry-After header
- 300-second cap on Retry-After values for safety
- Comprehensive test coverage for Retry-After parsing and integration

### Changed
- Enhanced cooldown logic to respect server Retry-After headers (takes precedence over default)
- HTTPStatusError handling now extracts Retry-After header for smarter backoff

## [0.2.0] - 2026-10-31

### Added
- Exponential backoff with jitter for retries
- endpoint_health() method for health summary
- py.typed marker for type checking support

### Changed
- Enhanced client with retry mechanism