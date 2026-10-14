#!/usr/bin/env bash

# Use $1 if passed, otherwise default to "obj"
TARGET_DIR="${1:-obj}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

shopt -s nullglob
files=("$TARGET_DIR"/*.7z)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .7z files found in '$TARGET_DIR'."
    exit 0
fi

for archive in "${files[@]}"; do
    base_name="$(basename "$archive" .7z)"
    out_dir="$TARGET_DIR/$base_name"

    echo "Extracting: $archive -> $out_dir"
    mkdir -p "$out_dir"
    
    # 7z is the modern official 7-zip CLI on Arch/Manjaro
    7z x "$archive" -o"$out_dir" -y
done

echo "All archives processed."