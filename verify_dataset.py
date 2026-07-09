"""
verify_dataset.py
==================

Dataset verification script for the "Road Accident Severity Prediction
using Machine Learning" project.

Purpose
-------
This script performs a STRUCTURAL VERIFICATION ONLY of the raw dataset
files. It checks for file existence, reports file size, and reads each
CSV file (safely) to report the number of rows and columns.

IMPORTANT — Scope Restrictions
-------------------------------
This script does NOT:
    - Clean the data
    - Modify the data
    - Merge datasets
    - Perform any EDA or feature engineering

It ONLY verifies that the dataset files are present, readable, and
structurally valid.

Usage
-----
    python verify_dataset.py

Author
------
Research Intern, IIIT Vadodara — On-Campus Research Internship
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_DIRECTORY: Path = Path("Dataset") / "raw"

EXPECTED_FILES: List[str] = [
    "Accident_Information.csv",
    "Vehicle_Information.csv",
]

# Encodings to attempt, in order, when reading each CSV file. UK Road Safety
# dataset exports are commonly encoded in UTF-8 or Latin-1 (ISO-8859-1).
CANDIDATE_ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class DatasetVerificationResult:
    """
    Container for the verification results of a single dataset file.

    Attributes:
        file_name (str): Name of the dataset file.
        exists (bool): Whether the file exists on disk.
        file_size_mb (Optional[float]): File size in megabytes.
        num_rows (Optional[int]): Number of rows read from the file.
        num_columns (Optional[int]): Number of columns read from the file.
        encoding_used (Optional[str]): Encoding that successfully read the file.
        error_message (Optional[str]): Error message, if verification failed.
    """

    file_name: str
    exists: bool = False
    file_size_mb: Optional[float] = None
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    encoding_used: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Return True if the file was verified without any errors."""
        return (
            self.exists
            and self.error_message is None
            and self.num_rows is not None
            and self.num_columns is not None
        )


# -----------------------------------------------------------------------------
# Core Verification Logic
# -----------------------------------------------------------------------------


def get_file_size_mb(file_path: Path) -> float:
    """
    Calculate the size of a file in megabytes.

    Args:
        file_path (Path): Path to the file.

    Returns:
        float: File size in megabytes, rounded to 2 decimal places.
    """
    size_bytes: int = os.path.getsize(file_path)
    size_mb: float = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def read_csv_safely(file_path: Path) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Attempt to read a CSV file using a series of candidate encodings,
    catching and reporting any errors encountered along the way.

    This function reads the FULL file only to determine shape (rows and
    columns). No data is modified, cleaned, or transformed.

    Args:
        file_path (Path): Path to the CSV file to read.

    Returns:
        tuple:
            - Optional[pd.DataFrame]: The loaded dataframe, or None on failure.
            - Optional[str]: The encoding that successfully read the file, or None.
            - Optional[str]: An error message, or None if successful.
    """
    last_error: Optional[str] = None

    for encoding in CANDIDATE_ENCODINGS:
        try:
            dataframe: pd.DataFrame = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False,
            )
            return dataframe, encoding, None

        except UnicodeDecodeError as decode_error:
            # Encoding mismatch — try the next candidate encoding.
            last_error = f"UnicodeDecodeError with encoding '{encoding}': {decode_error}"
            continue

        except pd.errors.EmptyDataError as empty_error:
            # File exists but contains no data.
            return None, None, f"EmptyDataError: {empty_error}"

        except pd.errors.ParserError as parser_error:
            # Malformed CSV structure.
            return None, None, f"ParserError: {parser_error}"

        except FileNotFoundError as file_error:
            # Should not normally occur here since existence is pre-checked,
            # but handled defensively regardless.
            return None, None, f"FileNotFoundError: {file_error}"

        except Exception as unexpected_error:  # pylint: disable=broad-except
            # Catch-all safeguard for any unforeseen read errors.
            return None, None, f"UnexpectedError: {unexpected_error}"

    # If we reach here, all candidate encodings failed with UnicodeDecodeError.
    return None, None, last_error


def verify_single_file(directory: Path, file_name: str) -> DatasetVerificationResult:
    """
    Verify a single dataset file: check existence, size, and shape.

    Args:
        directory (Path): Directory in which the file should reside.
        file_name (str): Name of the file to verify.

    Returns:
        DatasetVerificationResult: The result of the verification process.
    """
    file_path: Path = directory / file_name
    result = DatasetVerificationResult(file_name=file_name)

    # Step 1: Check file existence.
    if not file_path.exists():
        result.exists = False
        result.error_message = f"File not found at expected path: {file_path}"
        return result

    result.exists = True

    # Step 2: Compute file size.
    try:
        result.file_size_mb = get_file_size_mb(file_path)
    except OSError as os_error:
        result.error_message = f"OSError while reading file size: {os_error}"
        return result

    # Step 3: Attempt to read the CSV file safely.
    dataframe, encoding_used, read_error = read_csv_safely(file_path)

    if read_error is not None:
        result.error_message = read_error
        return result

    if dataframe is not None:
        result.num_rows = dataframe.shape[0]
        result.num_columns = dataframe.shape[1]
        result.encoding_used = encoding_used

    return result


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_verification_report(results: List[DatasetVerificationResult]) -> None:
    """
    Print a clean, professional, human-readable verification report to
    the console.

    Args:
        results (List[DatasetVerificationResult]): List of verification results.
    """
    separator: str = "=" * 60
    print(separator)
    print("DATASET VERIFICATION REPORT")
    print(separator)

    for result in results:
        print(f"\nFile Name        : {result.file_name}")

        if not result.exists:
            print("Status           : NOT FOUND")
            print(f"Details          : {result.error_message}")
            continue

        print(f"File Size (MB)   : {result.file_size_mb}")

        if result.error_message:
            print("Status           : FAILED TO LOAD")
            print(f"Error            : {result.error_message}")
            continue

        print(f"Number of Rows   : {result.num_rows}")
        print(f"Number of Columns: {result.num_columns}")
        print(f"Encoding Used    : {result.encoding_used}")
        print("Status           : Successfully Loaded")

    print(f"\n{separator}")

    total_files: int = len(results)
    successful_files: int = sum(1 for r in results if r.is_successful)

    print(f"Summary          : {successful_files}/{total_files} file(s) verified successfully")
    print(separator)


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main() -> None:
    """
    Main execution function.

    Verifies that both expected dataset files exist in the raw dataset
    directory, reports their size and shape, and prints a summary report.
    Exits with a non-zero status code if any file fails verification.
    """
    if not DATASET_DIRECTORY.exists():
        print(
            f"[ERROR] Dataset directory not found: {DATASET_DIRECTORY}\n"
            f"Please ensure the dataset is placed at: {DATASET_DIRECTORY.resolve()}"
        )
        sys.exit(1)

    results: List[DatasetVerificationResult] = [
        verify_single_file(DATASET_DIRECTORY, file_name) for file_name in EXPECTED_FILES
    ]

    print_verification_report(results)

    all_successful: bool = all(result.is_successful for result in results)

    if not all_successful:
        sys.exit(1)


if __name__ == "__main__":
    main()
