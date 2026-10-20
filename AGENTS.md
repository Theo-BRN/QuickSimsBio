# AGENTS.md

Guidance for AI coding agents working in this repository.

**Read [DEVELOPMENT.md](DEVELOPMENT.md) first.** It is the single source of truth for the package's design decisions, architecture, naming conventions, tests and known issues. [README.md](README.md) covers usage. Keep those files up to date rather than duplicating their content here.

## Quick reference

```bash
poetry install                              # install dependencies
poetry run python -c "import quicksimsbio"  # smoke test
poetry run pytest                           # regression tests
```

- Source: `src/quicksimsbio/` — `sampling.py` (core) and `curve_fitting.py` (standalone).
- Public API: the five functions exported from `__init__.py`. Everything else is internal.
- Pass `model=dm` explicitly to basico calls rather than relying on the globally loaded model.
