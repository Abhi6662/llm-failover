# llm-failover

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python library for OpenAI-compatible multi-endpoint failover with automatic cooldowns and SQLite telemetry.

## Features

- **Automatic Failover**: Rotates across multiple endpoints on 429/502/503 errors or timeouts
- **Cooldown Tracking**: Per-endpoint cooldown prevents hammering failing services
- **SQLite Telemetry**: Tracks token usage and latency for cost monitoring
- **OpenAI Compatible**: Drop-in replacement for standard OpenAI clients

## Installation

```bash
pip install llm-failover
```

## Usage

```python
from llm_failover import FailoverClient, EndpointConfig, Telemetry

endpoints = [
    EndpointConfig(url="https://api.primary.com/v1", api_key="sk-xxx"),
    EndpointConfig(url="https://api.backup.com/v1", api_key="sk-yyy", cooldown_seconds=30)
]

telemetry = Telemetry()
client = FailoverClient(endpoints=endpoints, telemetry=telemetry)

response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
    max_tokens=50
)

print(response["choices"][0]["message"]["content"])
print(telemetry.get_stats())
```

## Development

```bash
poetry install
poetry run pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.
