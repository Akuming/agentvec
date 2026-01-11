/**
 * Simple test script for AgentVec WASM bindings.
 * Run with: node test.mjs
 */

import { AgentVec, Metric } from './pkg/agentvec_js.js';

console.log('Testing AgentVec WASM bindings...\n');

// Create database
console.log('1. Creating database...');
const db = new AgentVec('./test_memory');
console.log('   ✓ Database created\n');

// Create collection
console.log('2. Creating collection...');
const memories = db.collection('episodic', 3, Metric.Cosine);
console.log(`   ✓ Collection "${memories.name}" created (${memories.dimensions}D, ${memories.metric === Metric.Cosine ? 'Cosine' : 'Other'} metric)\n`);

// Add vectors
console.log('3. Adding vectors...');
const id1 = memories.add(new Float32Array([1.0, 0.0, 0.0]), { type: 'user', name: 'alice' }, null, null);
const id2 = memories.add(new Float32Array([0.0, 1.0, 0.0]), { type: 'user', name: 'bob' }, null, null);
const id3 = memories.add(new Float32Array([0.9, 0.1, 0.0]), { type: 'assistant', name: 'claude' }, null, null);
console.log(`   ✓ Added 3 vectors: ${id1.slice(0, 8)}..., ${id2.slice(0, 8)}..., ${id3.slice(0, 8)}...\n`);

// Check length
console.log('4. Checking collection length...');
console.log(`   ✓ Collection has ${memories.length} vectors\n`);

// Search
console.log('5. Searching for similar vectors to [1.0, 0.0, 0.0]...');
const results = memories.search(new Float32Array([1.0, 0.0, 0.0]), 3, null);
console.log('   Results:');
for (const result of results) {
    console.log(`   - ${result.id.slice(0, 8)}... (score: ${result.score.toFixed(4)}, metadata: ${result.metadataJson})`);
}
console.log();

// Search with filter
console.log('6. Searching with filter { type: "user" }...');
const filtered = memories.search(new Float32Array([1.0, 0.0, 0.0]), 3, { type: 'user' });
console.log(`   Found ${filtered.length} results (expected 2):`);
for (const result of filtered) {
    console.log(`   - ${result.id.slice(0, 8)}... (score: ${result.score.toFixed(4)})`);
}
console.log();

// Get by ID
console.log('7. Getting vector by ID...');
const retrieved = memories.get(id1);
if (retrieved) {
    console.log(`   ✓ Retrieved: ${retrieved.id.slice(0, 8)}... with metadata: ${retrieved.metadataJson}\n`);
} else {
    console.log('   ✗ Failed to retrieve vector\n');
}

// Delete
console.log('8. Deleting a vector...');
const deleted = memories.delete(id2);
console.log(`   ✓ Deleted: ${deleted}\n`);

// Check new length
console.log('9. Checking collection length after delete...');
console.log(`   ✓ Collection now has ${memories.length} vectors\n`);

// Compact
console.log('10. Compacting collection...');
const stats = memories.compact();
console.log(`    ✓ Compacted: removed ${stats.tombstonesRemoved} tombstones in ${stats.durationMs}ms\n`);

// List collections
console.log('11. Listing collections...');
const collections = db.collections();
console.log(`    ✓ Collections: ${collections.join(', ')}\n`);

// Drop collection
console.log('12. Dropping collection...');
const dropped = db.dropCollection('episodic');
console.log(`    ✓ Dropped: ${dropped}\n`);

console.log('All tests passed! 🎉');
