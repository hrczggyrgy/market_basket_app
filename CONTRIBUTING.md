# Contributing to Market Basket Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows a standard open source Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- No harassment, discrimination, or offensive behavior

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- A GitHub account

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/market_basket_app.git
   cd market_basket_app
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate  # Windows
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # If pyproject.toml has dev extras
   ```

5. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions

### Commit Messages

Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(rules): add chi-squared metric to association rules
fix(cdt): resolve dendrogram rendering issue
docs(readme): update installation instructions
```

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following code standards

3. Run tests and linting:
   ```bash
   ruff check src/
   ruff format src/
   pytest tests/ -v  # if tests exist
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat(module): description of changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request

## Code Standards

### Python Style

- Follow PEP 8 (enforced by `ruff`)
- Type hints for all public functions
- Docstrings for all public modules, classes, and functions
- Maximum line length: 100 characters

### Imports

```python
# Standard library
import pandas as pd
import numpy as np

# Third-party
from sklearn.cluster import KMeans

# Local
from src.analytics.cohort import compute_cohorts
```

### Type Hints

```python
def compute_cohorts(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
    metric: str = "retention",
) -> pd.DataFrame:
    """Compute cohort analysis matrix."""
```

### Docstrings (Google Style)

```python
def compute_cohorts(
    transactions_df: pd.DataFrame,
    cohort_period: str = "M",
    metric: str = "retention",
) -> pd.DataFrame:
    """Compute cohort analysis matrix.

    Args:
        transactions_df: Transaction data with date, customer_id, etc.
        cohort_period: Period for cohort definition ('W', 'M', 'Q')
        metric: What to measure ('retention', 'revenue', 'orders')

    Returns:
        Cohort matrix with periods as columns, cohorts as rows

    Raises:
        ValueError: If metric is not supported.
    """
```

### Error Handling

- Use specific exceptions (`ValueError`, `KeyError`) not bare `Exception`
- Log errors appropriately
- Fail fast with clear messages

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cohort.py -v
```

### Test Structure

```
tests/
├── unit/              # Unit tests for individual functions
├── integration/       # Integration tests
├── fixtures/          # Test data fixtures
└── conftest.py        # Pytest configuration
```

### Writing Tests

```python
import pytest
import pandas as pd
from src.analytics.cohort import compute_cohorts

def test_compute_cohorts_retention():
    """Test retention cohort computation."""
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "customer_id": [f"C{i%10:03d}" for i in range(100)],
        "transaction_id": [f"INV{i:06d}" for i in range(100)],
        "stockcode": ["SKU001"] * 100,
        "product": ["Product A"] * 100,
        "price": [10.0] * 100,
        "quantity": [1] * 100,
    })

    result = compute_cohorts(df, cohort_period="M", metric="retention")

    assert isinstance(result, pd.DataFrame)
    assert "Period 0" in result.columns
    assert result.iloc[0, 0] == 1.0  # Period 0 retention is always 100%
```

## Documentation

### Updating Documentation

- Update `README.md` for new features
- Update docstrings for changed functions
- Add examples for new public APIs

### API Documentation

- Public functions must have docstrings
- Complex modules should have module-level docstrings
- Type hints serve as inline documentation

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (ruff passes)
- [ ] All tests pass
- [ ] Type hints added for new code
- [ ] Docstrings updated for public APIs
- [ ] No debug prints or commented code
- [ ] Commit messages are clear and follow convention

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactor

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Screenshots (if UI changes)
Add screenshots here

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
```

### Review Process

1. Automated checks must pass (lint, tests)
2. At least one maintainer review required
2. Address all review comments
3. Squash commits if requested
4. Merge after approval

## Questions?

Open an issue or start a discussion on GitHub if you have questions about contributing.