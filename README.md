# Data Drift Detector

This package contains some developmental tools to detect and compare differences between 2 datasets, with the primary intention of detecting data drift.

Data Drift Detector contains a class `DataDriftDetector` which takes in 2 pandas dataframes with the same columns and provides a few useful methods to compare and analyze the differences between the 2 datasets.

## Installation
Install the package with pip

    pip install data-drift-Detector

## Example Usage

To compare 2 datasets:

    from data_drift_detector import DataDriftDetector

    detector = DataDriftDetector(df_1, df_2)

You may also view an example notebook in the following directory `examples/example_usage.ipynb`.
