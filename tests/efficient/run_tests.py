#!/usr/bin/env python
"""Efficient test runner for the consolidated test suite.

Usage:
    python tests/efficient/run_tests.py              # Run all efficient tests
    python tests/efficient/run_tests.py --core       # Run core analytics tests only
    python tests/efficient/run_tests.py --ui         # Run UI component tests only
    python tests/efficient/run_tests.py --fast       # Run fast tests only (skip slow)
    python tests/efficient/run_tests.py --coverage   # Run with coverage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(args: argparse.Namespace) -> int:
    """Run pytest with appropriate arguments."""
    test_dir = Path(__file__).parent
    
    # Base pytest command
    cmd = [
        "python", "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
    ]
    
    # Add test selection
    if args.core:
        cmd.extend(["-k", "not ui and not insights and not opportunities"])
    elif args.ui:
        cmd.extend(["-k", "ui or insights or opportunities"])
    elif args.fast:
        cmd.extend(["-k", "not slow and not e2e"])
    
    # Coverage
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    
    # Parallel execution
    if not args.no_parallel:
        cmd.extend(["-n", "auto"])
    
    # Additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Efficient test runner")
    parser.add_argument("--core", action="store_true", help="Run core analytics tests only")
    parser.add_argument("--ui", action="store_true", help="Run UI component tests only")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only (skip slow/e2e)")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel execution")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Additional pytest arguments")
    
    args = parser.parse_args()
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())