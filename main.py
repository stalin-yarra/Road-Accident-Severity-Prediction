"""
main.py
=======

Entry point script for the "Road Accident Severity Prediction using Machine
Learning" project.

Purpose
-------
This script exists solely to verify that the project has been initialized
correctly (Phase 1: Project Initialization). It performs NO data loading,
preprocessing, EDA, feature engineering, or model training.

Usage
-----
    python main.py

Author
------
Research Intern, IIIT Vadodara — On-Campus Research Internship
"""

from __future__ import annotations

import sys
from datetime import datetime


def print_banner(message: str, width: int = 42) -> None:
    """
    Print a formatted banner message surrounded by separator lines.

    Args:
        message (str): The message to display inside the banner.
        width (int): The width of the separator line. Defaults to 42.
    """
    separator: str = "-" * width
    print(separator)
    print(message)
    print(separator)


def get_initialization_timestamp() -> str:
    """
    Generate a human-readable timestamp representing the moment the
    project initialization check was executed.

    Returns:
        str: A formatted timestamp string (YYYY-MM-DD HH:MM:SS).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def verify_python_version(minimum_major: int = 3, minimum_minor: int = 9) -> bool:
    """
    Verify that the current Python interpreter meets the minimum
    version requirement for this project.

    Args:
        minimum_major (int): Minimum required major version.
        minimum_minor (int): Minimum required minor version.

    Returns:
        bool: True if the Python version is sufficient, False otherwise.
    """
    current_version = sys.version_info
    return (current_version.major, current_version.minor) >= (
        minimum_major,
        minimum_minor,
    )


def main() -> None:
    """
    Main execution function.

    Verifies that the project environment is initialized correctly and
    prints a confirmation banner to the console.
    """
    try:
        print_banner("Road Accident Severity Prediction")
        print("Project Initialized Successfully")
        print(f"Timestamp        : {get_initialization_timestamp()}")
        print(
            f"Python Version   : "
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )

        if not verify_python_version():
            print(
                "\n[WARNING] This project recommends Python 3.9 or higher. "
                "Please consider upgrading your Python interpreter."
            )

        print("-" * 42)

    except Exception as error:  # pylint: disable=broad-except
        # Catch-all safeguard: initialization check should never crash
        # ungracefully. Any unexpected error is reported clearly.
        print(f"[ERROR] Project initialization check failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
