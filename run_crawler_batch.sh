#!/bin/bash

# Batch crawler script - processes each package individually
# Usage: ./run_crawler_batch.sh pkgs.txt

PKGS_FILE="$1"

if [ -z "$PKGS_FILE" ]; then
    echo "Usage: $0 <pkgs_file>"
    echo "  pkgs_file: Text file with one package name per line"
    exit 1
fi

if [ ! -f "$PKGS_FILE" ]; then
    echo "Error: File '$PKGS_FILE' not found"
    exit 1
fi

# Get pipeline arguments from command line (everything after pkgs_file)
PIPELINE_ARGS="${@:2}"

# Process each package
TOTAL=$(wc -l < "$PKGS_FILE")
FAILED=0

echo "Processing $TOTAL packages..."
echo "================================"

while IFS= read -r pkg || [ -n "$pkg" ]; do
    # Skip empty lines
    [ -z "$pkg" ] && continue
    
    echo ""
    echo "[Package $pkg] Starting crawler..."
    echo "================================"
    
    if python main.py -pkg "$pkg" $PIPELINE_ARGS; then
        echo "[Package $pkg] ✓ Completed successfully"
    else
        echo "[Package $pkg] ✗ FAILED"
        ((FAILED++))
    fi
    echo ""
    
done < "$PKGS_FILE"

echo "================================"
echo "Summary: $((TOTAL - FAILED))/$TOTAL packages completed successfully"
[ $FAILED -gt 0 ] && echo "Failed: $FAILED"
