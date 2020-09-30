# Data Drift Detector

This package contains some developmental tools to detect and compare differences between 2 datasets, with the primary intention of detecting data drift.

Data Drift Detector contains a class `DataDriftDetector` which takes in 2 pandas dataframes with the same columns and provides a few useful methods to compare and analyze the differences between the 2 datasets.

## Installation
Install the package with pip

    pip install data-drift-detector

## Example Usage

To compare 2 datasets:

    from data_drift_detector import DataDriftDetector

    detector = DataDriftDetector(df_prior = df_1, df_post = df_2)
    detector.calculate_drift()
    detector.plot_numeric_to_numeric()
    detector.plot_categorical_to_numeric()
    detector.compare_ml_efficacy(target_column="some_target")

You may also view an example notebook in the following directory `examples/example_usage.ipynb`.
