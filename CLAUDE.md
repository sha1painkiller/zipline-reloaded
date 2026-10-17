# Zipline Reloaded - Project Context for Claude

## Overview
Zipline is a Pythonic algorithmic trading library for backtesting trading strategies.
This repository is a personal fork of Stefan Jansen's zipline-reloaded, which is
unmaintained. **The fork is the source of truth**: its only remote is `origin`
(github.com/sha1painkiller/zipline-reloaded), and it is installed editable into the
`us-strategies-zlr` venv, so changes here change those backtests and live signals.

The fork diverges from upstream (last upstream commit `943010b9`) in two ways:
- **pandas 3 support** (also pandas 2.2+/numpy 2.x compatibility fixes).
- **Simulation-correctness patches** — slippage overfill, daily-mode EODCancel,
  dividend pay-date handling, splits on non-trading days, history adjustment
  merging, MaxDrawdown, pipeline masking, before_trading_start anchoring, and more.
  Do not "restore" upstream behaviour in these areas; each fix's reasoning is in
  its commit message. `git log --oneline 943010b9..main` lists them.

## Key Information
- **Python Version**: >= 3.10
- **Main Dependencies**: pandas >= 2.2 (runs on 3.0), SQLAlchemy >= 2.0, numpy >= 2.0
- **Documentation**: https://zipline.ml4trading.io
- **Community**: https://exchange.ml4trading.io

## Project Structure
- `src/zipline/`: Main source code
  - `algorithm.py`: Core algorithm execution
  - `api.py`: Public API functions
  - `data/`: Data ingestion and handling
  - `finance/`: Financial calculations and order execution
  - `pipeline/`: Factor-based screening system
- `tests/`: Test suite
- `docs/`: Documentation source

## Development Commands
```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_algorithm.py

# Build documentation
cd docs && make html

# Install in development mode
pip install -e .

# Rebuild the Cython extensions after editing any .pyx/.pxd file — the editable
# install imports the compiled .so, so a .pyx edit does nothing until rebuilt
python setup.py build_ext --inplace
```

Regression tests for the fork's own simulation fixes:
`tests/test_simulation_correctness.py` and `tests/finance/test_dividends.py`.
Run them (plus the area you touched) before pushing any change to `main`.

## Testing Approach
- Unit tests use pytest
- Test data is stored in `tests/resources/`
- Mock trading environments for testing strategies

## Common Tasks
1. **Implementing new data bundles**: See `src/zipline/data/bundles/`
2. **Adding new pipeline factors**: See `src/zipline/pipeline/factors/`
3. **Modifying order execution**: See `src/zipline/finance/execution.py`
4. **Working with trading calendars**: Uses `exchange_calendars` library

## Current Branch
Work happens on `main`, which tracks `origin/main`. The former `pandas3-compat` and
`bugfixes/simulation-correctness` branches are merged into it and deleted.
`us-strategies-zlr/requirements.txt` pins a commit SHA of this fork for its Docker
build, so a change only reaches production after that pin is bumped.

## Important Notes
- The project uses Cython for performance-critical components
- Be careful with numpy/pandas API changes due to major version updates
- Trading calendars are handled by the external `exchange_calendars` package
