# Copyright (c) Microsoft. All rights reserved

"""Script to generate Jupyter notebooks from marimo notebooks."""

import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    """Generate Jupyter notebooks from marimo .py files."""
    demos_dir = Path(__file__).parents[1] / "demos"

    marimo_notebooks = sorted(demos_dir.glob("*.py"))
    marimo_notebooks = [nb for nb in marimo_notebooks if nb.name != "generate_ipynbs.py"]

    if not marimo_notebooks:
        print("No marimo notebooks found in demos/")
        return

    print(f"Found {len(marimo_notebooks)} marimo notebook(s) to convert")

    for notebook_path in marimo_notebooks:
        output_path = notebook_path.with_suffix(".ipynb")
        print(f"\nExporting {notebook_path.name} to {output_path.name}")

        try:
            # Export using marimo
            subprocess.run(
                ["uv", "run", "marimo", "export", "ipynb", str(notebook_path), "-o", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Execute the notebook to capture outputs
            print(f"  Executing {output_path.name} to capture outputs...")
            subprocess.run(
                ["jupyter", "execute", "--inplace", "--timeout=1000", "--allow-errors", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            print("  ✓ Executed successfully")

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error: {e}")
            if e.stdout:
                print(f"    stdout: {e.stdout}")
            if e.stderr:
                print(f"    stderr: {e.stderr}")
            sys.exit(1)

    print(f"\n✓ Successfully generated {len(marimo_notebooks)} Jupyter notebook(s)")
    print("\nGenerated notebooks:")
    for notebook_path in marimo_notebooks:
        output_path = notebook_path.with_suffix(".ipynb")
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  - {output_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    # Check if required tools are available
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'uv' is not installed or not in PATH")
        sys.exit(1)

    try:
        subprocess.run(["marimo", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'marimo' is not installed. Run 'uv pip install marimo'")
        sys.exit(1)

    try:
        subprocess.run(["jupyter", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'jupyter' is not installed. Run 'uv pip install nbclient'")
        sys.exit(1)

    # Check for OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Notebooks may fail to execute properly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    main()
