# AgentVec Mobile Bindings

Mobile bindings for iOS (Swift) and Android (Kotlin) using UniFFI.

## Overview

This crate provides native mobile bindings for the AgentVec vector database, allowing seamless integration with iOS and Android applications.

### Why This Works for Mobile

- **Lightweight**: Pure Rust with `redb` and `memmap2` - no SQLite conflicts on iOS
- **Memory Efficient**: Memory-mapped I/O perfect for mobile constraints
- **Zero Runtime Deps**: No dynamic dependencies required
- **ACID Guarantees**: Full transaction support with crash recovery
- **Native Performance**: Compiled to native code for each platform

## Building

### Prerequisites

```bash
# Install Rust targets for mobile
rustup target add aarch64-apple-ios        # iOS (ARM64)
rustup target add x86_64-apple-ios         # iOS Simulator
rustup target add aarch64-linux-android    # Android (ARM64)
rustup target add armv7-linux-androideabi  # Android (ARM)
```

### Generate Language Bindings

The bindings are generated using the built-in uniffi-bindgen tool:

```bash
# From workspace root

# Generate Swift bindings
cargo run -p agentvec-mobile --features bindgen --bin uniffi-bindgen -- generate agentvec-mobile/src/agentvec.udl --language swift --out-dir agentvec-mobile/bindings/swift

# Generate Kotlin bindings
cargo run -p agentvec-mobile --features bindgen --bin uniffi-bindgen -- generate agentvec-mobile/src/agentvec.udl --language kotlin --out-dir agentvec-mobile/bindings/kotlin
```

### Build for iOS

```bash
# Build for iOS device
cargo build -p agentvec-mobile --release --target aarch64-apple-ios

# Build for iOS simulator
cargo build -p agentvec-mobile --release --target x86_64-apple-ios
```

The generated files:
- `bindings/swift/agentvec.swift` - Swift interface
- `bindings/swift/agentvecFFI.h` - C header for FFI
- `bindings/swift/agentvecFFI.modulemap` - Module map for Xcode
- `target/aarch64-apple-ios/release/libagentvec_mobile.a` - Static library

### Build for Android

```bash
# Build for Android ARM64
cargo build -p agentvec-mobile --release --target aarch64-linux-android

# Build for Android ARM
cargo build -p agentvec-mobile --release --target armv7-linux-androideabi
```

The generated files:
- `bindings/kotlin/uniffi/agentvec/agentvec.kt` - Kotlin interface
- `target/aarch64-linux-android/release/libagentvec_mobile.so` - Shared library

## Usage

### Swift (iOS)

```swift
import agentvec

// Open database
let db = try AgentVec(path: "\(documentDir)/my_vectors.db")

// Create collection
let collection = try db.collection(
    name: "embeddings",
    dimensions: 384,
    metric: .cosine
)

// Add vectors
let vector: [Float] = Array(repeating: 0.1, count: 384)
let metadata = "{\"text\": \"Hello, world!\"}"
let id = try collection.add(
    vector: vector,
    metadataJson: metadata,
    id: nil,
    ttl: nil
)

// Search
let results = try collection.search(
    vector: vector,
    k: 10,
    whereJson: nil
)

for result in results {
    print("ID: \(result.id), Score: \(result.score)")
    print("Metadata: \(result.metadataJson)")
}

// Sync to disk
try collection.sync()
```

### Kotlin (Android)

```kotlin
import agentvec.*

// Open database
val db = AgentVec(filesDir.path + "/my_vectors.db")

// Create collection
val collection = db.collection(
    name = "embeddings",
    dimensions = 384u,
    metric = Metric.COSINE
)

// Add vectors
val vector = FloatArray(384) { 0.1f }.toList()
val metadata = "{\"text\": \"Hello, world!\"}"
val id = collection.add(
    vector = vector,
    metadataJson = metadata,
    id = null,
    ttl = null
)

// Search
val results = collection.search(
    vector = vector,
    k = 10u,
    whereJson = null
)

results.forEach { result ->
    println("ID: ${result.id}, Score: ${result.score}")
    println("Metadata: ${result.metadataJson}")
}

// Sync to disk
collection.sync()
```

## API Reference

### AgentVec

Main database interface.

- `open(path: String)` - Open or create database
- `collection(name: String, dimensions: UInt32, metric: Metric)` - Get or create collection
- `getCollection(name: String)` - Get existing collection
- `collections()` - List all collection names
- `dropCollection(name: String)` - Delete collection
- `sync()` - Flush all pending writes
- `recoveryStats()` - Get recovery statistics from last open

### Collection

Vector collection interface.

- `add(vector: [Float], metadataJson: String, id: String?, ttl: UInt64?)` - Add vector
- `upsert(id: String, vector: [Float], metadataJson: String, ttl: UInt64?)` - Insert or update
- `search(vector: [Float], k: UInt32, whereJson: String?)` - Find nearest neighbors
- `get(id: String)` - Get record by ID
- `delete(id: String)` - Delete record
- `compact()` - Compact collection and reclaim space
- `len()` - Get number of active records
- `preload()` - Preload vectors into memory
- `sync()` - Flush pending writes
- `exportToFile(path: String)` - Export to file
- `importFromFile(path: String)` - Import from file
- `dimensions()` - Get vector dimensions
- `name()` - Get collection name
- `metric()` - Get distance metric
- `vectorsSizeBytes()` - Get storage size

### Metric

Distance metrics for similarity.

- `Cosine` - Cosine similarity
- `Dot` - Dot product
- `L2` - Euclidean distance

### SearchResult

Result from vector search.

- `id: String` - Record ID
- `score: Float` - Similarity score
- `metadataJson: String` - Metadata as JSON string

### Filtering

Metadata filtering uses JSON syntax:

```json
{
  "category": "news",
  "timestamp": { "$gt": 1234567890 }
}
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`

## Integration

### iOS (Xcode)

1. Add the `.swift` file to your Xcode project
2. Add the `.a` library to "Frameworks and Libraries"
3. Add library search path to the directory containing the `.a` file

### Android (Gradle)

1. Add the `.kt` file to `app/src/main/java/`
2. Create `app/src/main/jniLibs/arm64-v8a/` directory
3. Copy `libagentvec_mobile.so` to the jniLibs directory
4. Repeat for other architectures (armeabi-v7a, etc.)

## Performance

Optimized for mobile constraints:

- **Small Footprint**: ~2MB binary size (release build, stripped)
- **Fast Queries**: Sub-millisecond searches on 10K vectors
- **Memory Efficient**: Memory-mapped storage, minimal heap usage
- **Battery Friendly**: No background threads, explicit sync control

## File Storage

### iOS

Store database in app's Documents directory:

```swift
let documentsPath = FileManager.default.urls(
    for: .documentDirectory,
    in: .userDomainMask
)[0].path
let dbPath = "\(documentsPath)/vectors.db"
```

### Android

Store in app's private files directory:

```kotlin
val dbPath = "${filesDir.path}/vectors.db"
```

## Thread Safety

- `AgentVec` and `Collection` are thread-safe
- Can be used from multiple threads concurrently
- Reads are lock-free, writes are synchronized

## License

MIT OR Apache-2.0
