#!/bin/bash
# filepath: /home/zazabap/Projects/Qforge/LogosQ/scripts/test.sh

# Exit on any error
set -e

echo "=== Starting comprehensive LogosQ testing ==="

# Step 1: Format code
echo "=== Formatting code ==="
cargo fmt --all -- --check
echo "✓ Code formatting verified"

# Step 2: Run clippy for linting
echo "=== Running Clippy linter ==="
cargo clippy --all-targets --all-features -- -D warnings
echo "✓ Clippy checks passed"

# Step 3: Run unit tests
echo "=== Running unit tests ==="
cargo test --all-features
echo "✓ Unit tests passed"

# Step 4: Run doc tests
echo "=== Running documentation tests ==="
cargo test --doc
echo "✓ Documentation tests passed"

# Step 5: Run benchmarks (if available)
if [ -d "benches" ] || grep -q "\[\[bench\]\]" Cargo.toml; then
    echo "=== Running benchmarks ==="
    cargo bench
    echo "✓ Benchmarks completed"
fi

# Step 6: Check for unused dependencies
echo "=== Checking for unused dependencies ==="
cargo +nightly udeps --all-targets 2>/dev/null || echo "! cargo-udeps not available (install with 'cargo install cargo-udeps')"

# Step 7: Generate and validate documentation
echo "=== Generating documentation ==="
cargo doc --no-deps
echo "✓ Documentation generated successfully"

# Step 8: Run coverage (if tarpaulin is installed)
echo "=== Running code coverage ==="
if command -v cargo-tarpaulin >/dev/null 2>&1; then
    cargo tarpaulin --out Html --output-dir coverage
    echo "✓ Coverage report generated in coverage/"
else
    echo "! cargo-tarpaulin not installed (install with 'cargo install cargo-tarpaulin')"
fi

# Step 9: Run examples
echo "=== Testing examples ==="
find examples -name "*.rs" | while read -r example; do
    echo "Running example: $example"
    cargo run --example $(basename "$example" .rs)
done
echo "✓ All examples successfully executed"

# Step 10: Build in release mode
echo "=== Building in release mode ==="
cargo build --release
echo "✓ Release build successful"

echo "=== All tests completed successfully! ==="