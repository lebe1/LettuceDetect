#!/usr/bin/env python
"""Script to run pytest tests for the lettucedetect package."""

import sys

import pytest


def run_tests():
    """Run pytest tests for the lettucedetect package."""
    # Run pytest with specified arguments
    args = [
        "-v",  # verbose output
        "--tb=short",  # shorter traceback format
        "tests/test_inference_pytest.py",  # only run inference tests
    ]

    # Add any command line arguments
    args.extend(sys.argv[1:])

    # Run pytest and return the exit code
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(run_tests())
