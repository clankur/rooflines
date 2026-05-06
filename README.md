# Rooflines

Interactive roofline model analysis for ML accelerators. Built with [marimo](https://marimo.io/).

**[Live demo](https://clankur.github.io/rooflines/)**

## Features

- Fetch model specs from HuggingFace (trending models pre-loaded)
- Compare chips side-by-side (H100, TPU v5e, etc.)
- Compute/memory/network regime analysis with configurable TP/DP mesh

## Development

```bash
make dev        # launch marimo editor
make build      # export WASM to docs/
make clean      # remove docs/
```

## Deploy

The site is deployed to GitHub Pages from the `docs/` directory on `main`. Push to `main` to deploy:

```bash
make build
git add docs/
git commit -m "rebuild site"
git push
```

Ensure GitHub Pages is configured to serve from `main` branch, `/docs` folder in the repo settings.
