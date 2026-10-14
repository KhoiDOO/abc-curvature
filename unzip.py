#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run unzip.sh to extract .7z files from a target directory."
    )
    parser.add_argument(
        "-d", "--directory",
        dest="target_dir",
        nargs="?",
        default="obj",
        help="Directory containing .7z archives (default: 'obj')",
    )
    args = parser.parse_args()

    # Locate unzip.sh relative to this script
    script_dir = Path(__file__).resolve().parent
    bash_script = script_dir / "unzip.sh"

    if not bash_script.is_file():
        sys.exit(f"Error: Script '{bash_script}' not found.")

    cmd = ["bash", str(bash_script), args.target_dir]

    try:
        # Run and stream output directly to stdout/stderr
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit("\nOperation cancelled by user.")


if __name__ == "__main__":
    main()