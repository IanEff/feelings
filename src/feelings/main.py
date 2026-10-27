"""
Main module for feelings.

A vector rendering of tenderness
"""

import argparse
import sys
from pathlib import Path


def greet(name: str) -> str:
    """
    Generate a greeting message.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"


def process_items(items: list[int]) -> dict[str, int | float]:
    """
    Process a list of items and return statistics.

    Args:
        items: List of integers to process

    Returns:
        Dictionary with statistics about the items
    """
    if not items:
        return {
            "count": 0,
            "sum": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
        }

    return {
        "count": len(items),
        "sum": sum(items),
        "mean": sum(items) / len(items),
        "min": min(items),
        "max": max(items),
    }


def main() -> int:
    """
    Main entry point for feelings.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="A vector rendering of tenderness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="World",
        help="Name to greet (default: World)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        print("Starting feelings...")
        print(f"Working directory: {Path.cwd()}")
        print(f"Python version: {sys.version.split()[0]}")
        print()

    # Main logic
    message = greet(args.name)
    print(message)

    # Example: Process some data
    sample_data = [1, 2, 3, 4, 5, 10, 20, 30]
    stats = process_items(sample_data)

    if args.verbose:
        print(f"\nSample data: {sample_data}")
        print("Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
