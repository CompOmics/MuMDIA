"""
Test runner script for MuMDIA test suite.

This script provides convenient ways to run the test suite with different
configurations and reporting options.
"""

import argparse
import sys
from pathlib import Path

import pytest


def run_tests(
    test_path=None,
    verbose=False,
    coverage=False,
    markers=None,
    failed_first=False,
    stop_on_first_fail=False,
):
    """
    Run the MuMDIA test suite with specified options.

    Args:
        test_path: Specific test file or directory to run
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        markers: Pytest markers to filter tests
        failed_first: Run previously failed tests first
        stop_on_first_fail: Stop on first test failure
    """
    # Base pytest arguments
    args = []

    # Add test path
    if test_path:
        args.append(str(test_path))
    else:
        args.append("tests/")

    # Verbose output
    if verbose:
        args.append("-v")

    # Coverage reporting
    if coverage:
        args.extend(
            [
                "--cov=feature_generators",
                "--cov=data_structures",
                "--cov=utilities",
                "--cov=parsers",
                "--cov=prediction_wrappers",
                "--cov=peptide_search",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    # Filter by markers
    if markers:
        args.extend(["-m", markers])

    # Failed first
    if failed_first:
        args.append("--lf")

    # Stop on first failure
    if stop_on_first_fail:
        args.append("-x")

    # Add useful options
    args.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Fail on unknown markers
            "--disable-warnings",  # Reduce noise from warnings
        ]
    )

    print(f"Running pytest with arguments: {' '.join(args)}")
    return pytest.main(args)


def run_specific_tests():
    """Run specific test categories."""
    test_categories = {
        "fragment": "tests/test_features_fragment_intensity.py",
        "data": "tests/test_data_structures.py",
        "utils": "tests/test_utilities.py",
        "all": "tests/",
    }

    print("Available test categories:")
    for category, path in test_categories.items():
        print(f"  {category}: {path}")

    return test_categories


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run MuMDIA test suite")

    parser.add_argument(
        "test_category",
        nargs="?",
        default="all",
        help="Test category to run (fragment, data, utils, all)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Enable coverage reporting"
    )

    parser.add_argument("--markers", "-m", help="Run tests with specific markers")

    parser.add_argument(
        "--failed-first",
        "-lf",
        action="store_true",
        help="Run previously failed tests first",
    )

    parser.add_argument(
        "--stop-on-fail", "-x", action="store_true", help="Stop on first test failure"
    )

    parser.add_argument(
        "--list-categories", action="store_true", help="List available test categories"
    )

    args = parser.parse_args()

    if args.list_categories:
        run_specific_tests()
        return 0

    # Get test categories
    test_categories = run_specific_tests()

    # Validate test category
    if args.test_category not in test_categories:
        print(f"Invalid test category: {args.test_category}")
        print("Use --list-categories to see available options")
        return 1

    test_path = test_categories[args.test_category]

    # Check if test path exists
    if not Path(test_path).exists():
        print(f"Test path does not exist: {test_path}")
        return 1

    # Run tests
    exit_code = run_tests(
        test_path=test_path,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers,
        failed_first=args.failed_first,
        stop_on_first_fail=args.stop_on_fail,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
