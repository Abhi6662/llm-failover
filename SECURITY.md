# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in llm-failover, please report it by:

1. **Do not** open a public issue
2. Email the maintainer or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue promptly.

## Security Considerations

- llm-failover manages API keys for multiple LLM endpoints
- Store API keys securely using environment variables or secret managers
- The library does not log API keys or sensitive request/response bodies
- Failover logic may retry failed requests - ensure idempotency for non-GET operations
- Telemetry data is local-only by default