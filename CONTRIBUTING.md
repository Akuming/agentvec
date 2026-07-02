# Contributing to AgentVec

Thanks for your interest in improving AgentVec! This guide covers how to build,
test, and submit changes.

## Project layout

AgentVec is a Cargo workspace:

- `agentvec/` — the core database (pure Rust): storage, HNSW, quantization, filtering.
- `agentvec-python/` — Python bindings (PyO3 + maturin), published to PyPI as `agentvec`.
- `agentvec-cli/` — command-line tool.
- `agentvec-memory/`, `agentvec-mcp/`, `agentvec-cache/` — higher-level Python packages built on top.

## Prerequisites

- **Rust** (stable, >= 1.70) — install via [rustup](https://rustup.rs).
- **Python 3.8–3.13** for the bindings. **Note:** the pinned PyO3 version does
  not build against Python 3.14 yet, so use 3.13 or earlier to build/test the
  Python extension. [`uv`](https://github.com/astral-sh/uv) is a convenient way
  to get a 3.13 interpreter alongside a newer system Python.
- **maturin** and **numpy** for building/testing the bindings (`pip install maturin numpy`).

## Building

```bash
# Core library and CLI (Rust only)
cargo build --release -p agentvec -p agentvec-cli

# Python bindings — build into the active (3.13) virtualenv
cd agentvec-python
maturin develop --release
```

> The Python bindings are a PyO3 `extension-module` and are **not** built with a
> plain `cargo build -p agentvec-python` (an extension module doesn't link
> libpython standalone). Always use `maturin develop` / `maturin build`.

## Testing

```bash
# Correctness suites — run in RELEASE mode
cargo test -p agentvec --release --lib --test comprehensive_tests --test robustness_tests
```

> The `benchmark_*` integration tests assert wall-clock timings and are meant to
> run in release mode; they are performance gates, not correctness tests. Use
> `cargo bench` for benchmarking. When validating a change, rely on the lib unit
> tests plus the `comprehensive_tests` and `robustness_tests` suites above.

For the Python side, build with `maturin develop` and exercise the bindings from
Python (list and NumPy inputs, batch, search, filtering).

## Code style

- Run `cargo fmt` before committing.
- The core crate enables `#![warn(clippy::pedantic)]`; keep new code clippy-clean
  (`cargo clippy -p agentvec`).
- Match the surrounding code's naming, documentation, and idioms.

## On-disk format changes

The vector file carries a format `version`. If you change the on-disk layout or
checksum, **bump the version and keep older versions readable** — existing
`.avdb` databases must continue to open. See `agentvec/src/storage/vectors.rs`
and its backward-compatibility tests for the pattern.

## Submitting changes

1. Fork and create a feature branch.
2. Make your change with tests, and update `CHANGELOG.md` under `[Unreleased]`.
3. Ensure the correctness suites pass and clippy is clean.
4. Open a pull request describing the change and how you verified it.

## License

By contributing, you agree that your contributions will be dual-licensed under
the [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE) licenses, matching the
project.
