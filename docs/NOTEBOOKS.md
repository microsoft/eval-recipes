# Generating Jupyter Notebooks

The marimo `.py` files in `demos/` are the source of truth for our demo notebooks. We generate `.ipynb` files with executed outputs for GitHub preview and sharing.

## When to Run

- Before merging PRs that modify marimo notebooks
- After creating new demo notebooks
- When updating notebook outputs with latest library changes

## How to Run

```bash
uv run scripts/generate_ipynbs.py
```

This will:
1. Export all marimo notebooks to `.ipynb` format
2. Execute them to capture outputs
3. Save in the `demos/` directory
