//! Export and import functionality for backup and data portability.
//!
//! Uses NDJSON (Newline Delimited JSON) format:
//! - Line 1: Header with version, collection info
//! - Line 2+: Records with id, vector, metadata
//!
//! # Example
//!
//! ```ignore
//! // Export
//! collection.export_to_file("backup.ndjson")?;
//!
//! // Import
//! let stats = collection.import_from_file("backup.ndjson")?;
//! println!("Imported {} records", stats.imported);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{AgentVecError as Error, Result};

/// Export file format version.
pub const EXPORT_VERSION: u32 = 1;

/// Header for export files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportHeader {
    /// Format version.
    pub version: u32,
    /// Collection name.
    pub collection: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance metric ("cosine", "dot", "l2").
    pub metric: String,
    /// Number of records in export.
    pub record_count: usize,
    /// Export timestamp (Unix seconds).
    pub exported_at: u64,
}

/// A single exported record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord {
    /// Record ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata JSON.
    pub metadata: JsonValue,
    /// Creation timestamp (Unix seconds).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    /// TTL in seconds from creation (not absolute expiry).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<u64>,
}

/// Statistics from an import operation.
#[derive(Debug, Clone, Default)]
pub struct ImportStats {
    /// Number of records successfully imported.
    pub imported: usize,
    /// Number of records skipped (e.g., duplicates).
    pub skipped: usize,
    /// Number of records that failed to import.
    pub failed: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Export writer for streaming exports.
pub struct ExportWriter<W: Write> {
    writer: BufWriter<W>,
    record_count: usize,
}

impl<W: Write> ExportWriter<W> {
    /// Create a new export writer and write the header.
    pub fn new(
        writer: W,
        collection: &str,
        dimensions: usize,
        metric: &str,
        record_count: usize,
    ) -> Result<Self> {
        let mut writer = BufWriter::new(writer);

        let header = ExportHeader {
            version: EXPORT_VERSION,
            collection: collection.to_string(),
            dimensions,
            metric: metric.to_string(),
            record_count,
            exported_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        let header_json = serde_json::to_string(&header)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        writeln!(writer, "{}", header_json)
            .map_err(|e| Error::Io(e))?;

        Ok(Self {
            writer,
            record_count: 0,
        })
    }

    /// Write a record to the export.
    pub fn write_record(&mut self, record: &ExportRecord) -> Result<()> {
        let record_json = serde_json::to_string(record)
            .map_err(|e| Error::Serialization(e.to_string()))?;

        writeln!(self.writer, "{}", record_json)
            .map_err(|e| Error::Io(e))?;

        self.record_count += 1;
        Ok(())
    }

    /// Finish writing and flush the buffer.
    pub fn finish(mut self) -> Result<usize> {
        self.writer.flush().map_err(|e| Error::Io(e))?;
        Ok(self.record_count)
    }
}

/// Export reader for streaming imports.
pub struct ExportReader<R: BufRead> {
    reader: R,
    header: ExportHeader,
    line_number: usize,
}

impl<R: BufRead> ExportReader<R> {
    /// Create a new export reader and read the header.
    pub fn new(mut reader: R) -> Result<Self> {
        let mut header_line = String::new();
        reader.read_line(&mut header_line)
            .map_err(|e| Error::Io(e))?;

        let header: ExportHeader = serde_json::from_str(&header_line)
            .map_err(|e| Error::Deserialization(format!("Invalid export header: {}", e)))?;

        if header.version > EXPORT_VERSION {
            return Err(Error::InvalidFormat(format!(
                "Export version {} is newer than supported version {}",
                header.version, EXPORT_VERSION
            )));
        }

        Ok(Self {
            reader,
            header,
            line_number: 1,
        })
    }

    /// Get the export header.
    pub fn header(&self) -> &ExportHeader {
        &self.header
    }

    /// Read the next record, returning None at end of file.
    pub fn read_record(&mut self) -> Result<Option<ExportRecord>> {
        let mut line = String::new();
        let bytes = self.reader.read_line(&mut line)
            .map_err(|e| Error::Io(e))?;

        if bytes == 0 {
            return Ok(None); // EOF
        }

        self.line_number += 1;

        // Skip empty lines
        let line = line.trim();
        if line.is_empty() {
            return self.read_record();
        }

        let record: ExportRecord = serde_json::from_str(line)
            .map_err(|e| Error::Deserialization(format!(
                "Invalid record at line {}: {}", self.line_number, e
            )))?;

        Ok(Some(record))
    }
}

/// Export a collection to a file.
pub fn export_to_file<P: AsRef<Path>>(
    path: P,
    collection_name: &str,
    dimensions: usize,
    metric: &str,
    records: impl Iterator<Item = ExportRecord>,
    record_count: usize,
) -> Result<usize> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::Io(e))?;

    let mut writer = ExportWriter::new(
        file,
        collection_name,
        dimensions,
        metric,
        record_count,
    )?;

    for record in records {
        writer.write_record(&record)?;
    }

    writer.finish()
}

/// Import records from a file.
pub fn import_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<(ExportHeader, Vec<ExportRecord>)> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(e))?;

    let reader = BufReader::new(file);
    let mut export_reader = ExportReader::new(reader)?;

    let header = export_reader.header().clone();
    let mut records = Vec::with_capacity(header.record_count);

    while let Some(record) = export_reader.read_record()? {
        records.push(record);
    }

    Ok((header, records))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use serde_json::json;

    #[test]
    fn test_export_header_roundtrip() {
        let header = ExportHeader {
            version: EXPORT_VERSION,
            collection: "test".to_string(),
            dimensions: 384,
            metric: "cosine".to_string(),
            record_count: 100,
            exported_at: 1234567890,
        };

        let json = serde_json::to_string(&header).unwrap();
        let parsed: ExportHeader = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.collection, header.collection);
        assert_eq!(parsed.dimensions, header.dimensions);
    }

    #[test]
    fn test_export_record_roundtrip() {
        let record = ExportRecord {
            id: "test-id".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: json!({"key": "value"}),
            created_at: Some(1234567890),
            ttl: Some(3600),
        };

        let json = serde_json::to_string(&record).unwrap();
        let parsed: ExportRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, record.id);
        assert_eq!(parsed.vector, record.vector);
        assert_eq!(parsed.metadata, record.metadata);
    }

    #[test]
    fn test_export_record_optional_fields() {
        let record = ExportRecord {
            id: "test-id".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: json!({"key": "value"}),
            created_at: None,
            ttl: None,
        };

        let json = serde_json::to_string(&record).unwrap();
        // Optional fields should not appear in JSON
        assert!(!json.contains("created_at"));
        assert!(!json.contains("ttl"));

        let parsed: ExportRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, record.id);
        assert!(parsed.created_at.is_none());
        assert!(parsed.ttl.is_none());
    }

    #[test]
    fn test_export_writer_reader_roundtrip() {
        let mut buffer = Vec::new();

        // Write
        {
            let mut writer = ExportWriter::new(
                &mut buffer,
                "test_collection",
                4,
                "cosine",
                2,
            ).unwrap();

            writer.write_record(&ExportRecord {
                id: "id1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: json!({"name": "first"}),
                created_at: None,
                ttl: None,
            }).unwrap();

            writer.write_record(&ExportRecord {
                id: "id2".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                metadata: json!({"name": "second"}),
                created_at: Some(1000),
                ttl: Some(3600),
            }).unwrap();

            let count = writer.finish().unwrap();
            assert_eq!(count, 2);
        }

        // Read
        {
            let cursor = Cursor::new(&buffer);
            let mut reader = ExportReader::new(cursor).unwrap();

            let header = reader.header();
            assert_eq!(header.collection, "test_collection");
            assert_eq!(header.dimensions, 4);
            assert_eq!(header.metric, "cosine");
            assert_eq!(header.record_count, 2);

            let r1 = reader.read_record().unwrap().unwrap();
            assert_eq!(r1.id, "id1");
            assert_eq!(r1.metadata["name"], "first");

            let r2 = reader.read_record().unwrap().unwrap();
            assert_eq!(r2.id, "id2");
            assert_eq!(r2.ttl, Some(3600));

            let r3 = reader.read_record().unwrap();
            assert!(r3.is_none()); // EOF
        }
    }

    #[test]
    fn test_file_roundtrip() {
        let tmpdir = tempfile::TempDir::new().unwrap();
        let path = tmpdir.path().join("export.ndjson");

        let records = vec![
            ExportRecord {
                id: "a".to_string(),
                vector: vec![1.0, 2.0],
                metadata: json!({"x": 1}),
                created_at: None,
                ttl: None,
            },
            ExportRecord {
                id: "b".to_string(),
                vector: vec![3.0, 4.0],
                metadata: json!({"x": 2}),
                created_at: None,
                ttl: None,
            },
        ];

        // Export
        let exported = export_to_file(
            &path,
            "test",
            2,
            "cosine",
            records.into_iter(),
            2,
        ).unwrap();
        assert_eq!(exported, 2);

        // Import
        let (header, imported) = import_from_file(&path).unwrap();
        assert_eq!(header.collection, "test");
        assert_eq!(header.dimensions, 2);
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].id, "a");
        assert_eq!(imported[1].id, "b");
    }
}
