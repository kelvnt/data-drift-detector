# Data Drift Detector
[![PyPI version](https://badge.fury.io/py/data-drift-detector.svg)](https://badge.fury.io/py/data-drift-detector)

This package contains some developmental tools to detect and compare statistical differences between 2 structurally similar pandas dataframes. The intended purpose is to detect data drift - where the statistical properties of an input variable change over time.

We provide a class `DataDriftDetector` which takes in 2 pandas dataframes and provides a few useful methods to compare and analyze the differences between the 2 datasets.

## Installation
Install the package with pip
```bash
pip install data-drift-detector
```
## Example Usage

To compare 2 datasets:
```python
from data_drift_detector import DataDriftDetector

# initialize detector
detector = DataDriftDetector(df_prior = df_1, df_post = df_2)

# methods to compare and analyze differences
detector.calculate_drift()
detector.plot_numeric_to_numeric()
detector.plot_categorical_to_numeric()
detector.plot_categorical()
detector.compare_ml_efficacy(target_column="some_target_column")
```
You may also view an example notebook in the following directory `examples/example_usage.ipynb` to explore how it may be used.
