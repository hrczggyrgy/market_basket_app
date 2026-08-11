# Agent Guidelines for Market Basket App

## Setup
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install: `pip install -e .`
- Dev extras: `pip install -e .[dev]`

## Running the App
- Start: `streamlit run app.py`
- Data: expects CSV with columns: date, transaction_id, stockcode, product, customer_id, price, quantity (optional: category, brand, size, flavor/variant, promo_flag, is_online, cost)
- Sample data: `sample_data/sample_transactions.csv`

## Testing
- Run unit tests: `pytest tests/unit -x -q -k "not slow"`
- With coverage: `pytest tests/unit --cov=src --cov-report=term-missing`
- Specific module: `pytest tests/unit/test_pricing.py -v`

## Code Quality
- Lint: `ruff check .`
- Format: `ruff check . --fix`
- Typecheck: `mypy src`

## Project Structure
- Entrypoint: `app.py`
- Core analytics: `src/analytics/`
- UI tabs: `src/ui/tabs/`
- Shared plot helpers: `src/ui/plots.py`
- Data contracts: `src/analytics/schemas.py`
- Config: `pyproject.toml`

## Notes
- Uses Streamlit 1.59+, Python >=3.10
- Dependencies locked via pyproject.toml
- Validation via pydantic schemas