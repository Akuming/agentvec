# Changelog

All notable changes to AgentVec are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-06-28

### Added
- **NumPy array input.** `add`, `upsert`, `add_batch`, and `search` now accept
  NumPy `float32` arrays (fast path) and `float64` arrays. `add_batch` accepts a
  2-D `[n, dim]` array. Python lists still work — fully backward compatible.

### Changed
- **GIL is released during inserts and searches.** The core operations run under
  `allow_threads`, so other Python threads run concurrently (~3x measured search
  throughput across 8 threads).
- **Vector-file header checksum upgraded to CRC32** (format version 2). Files
  written by earlier releases (version 1) continue to verify with the legacy
  FNV-1a checksum, so existing `.avdb` databases open with no migration and no
  data loss.
- Sharpened the PyPI project description and added a proper long-description
  (README), authors, and homepage metadata.
- Replaced placeholder repository URLs with the real repository.

## [0.2.0]

### Added
- Metadata filter operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`,
  `$in`, `$nin`.
- Export/import for backup and transfer.
- Product quantization for vector compression.
- Mobile bindings (UniFFI).

## [0.1.1]

### Fixed
- CI/build fixes for the PyPI release workflow across supported Python versions.

## [0.1.0]

### Added
- Initial release: embedded vector database for AI agent memory.
- Memory-mapped vector storage with ACID metadata (redb).
- HNSW approximate search with parallel and incremental construction.
- TTL / memory decay, metadata filtering, and Python bindings (PyO3 + maturin).

[Unreleased]: https://github.com/Akuming/agentvec/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Akuming/agentvec/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Akuming/agentvec/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Akuming/agentvec/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Akuming/agentvec/releases/tag/v0.1.0
