"""
Pytest configuration for agentvec-mcp tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires real dependencies)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless --run-integration is passed."""
    if config.getoption("--run-integration", default=False):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (requires agentvec, agentvec-memory, and MCP)"
    )
